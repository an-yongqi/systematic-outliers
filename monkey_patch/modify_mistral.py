import math
import types
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint
from transformers.models.mistral.modeling_mistral import (
    apply_rotary_pos_emb, 
    repeat_kv,
)


def mistral_custom_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )

    if residual.device.index != hidden_states.device.index:
        residual = residual.to(hidden_states.device)
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    self.feat = hidden_states.clone().detach().cpu().double()

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

def enable_mistral_custom_decoderlayer(layer, layer_id):
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        mistral_custom_decoderlayer_forward, layer
    )


def mistral_custom_mlp_forward(self, x):
    inp_down = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    self.feat = inp_down.clone().detach().cpu().double()
    down_proj = self.down_proj(inp_down)
    return down_proj
    

def enable_mistral_custom_mlp(layer, layer_id):
    modified_module = layer.mlp
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(mistral_custom_mlp_forward, modified_module)

    return modified_module


def mistral_custom_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ###################################################
        # self.hidden_states = hidden_states.detach().cpu().clone()
        self.query_states = query_states.detach().cpu().clone()
        self.key_states = key_states.detach().cpu().clone()
        self.value_states = value_states.detach().cpu().clone()
        # ###################################################

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones((q_len, q_len), device=hidden_states.device)).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
        causal_mask = causal_mask.expand(bsz, 1, q_len, q_len)
        attn_weights = attn_weights + causal_mask

        # ###################################################
        self.attn_logits = attn_weights.detach().cpu().clone()
        # ###################################################

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # ###################################################
        self.attn_probs = attn_weights.detach().cpu().clone()
        # ###################################################

        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def enable_mistral_custom_attention(layer, layer_id):
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(mistral_custom_attention_forward, modified_module)

    return modified_module
