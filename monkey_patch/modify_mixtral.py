import math
import types
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint
from transformers.models.mixtral.modeling_mixtral import (
    apply_rotary_pos_emb, 
    repeat_kv,
)


def mixtral_custom_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

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
        cache_position=cache_position,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states, router_logits = self.block_sparse_moe(hidden_states)
    hidden_states = residual + hidden_states
    
    self.feat = hidden_states.clone().detach().cpu().double()

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_router_logits:
        outputs += (router_logits,)

    return outputs

def enable_mixtral_custom_decoderlayer(layer, layer_id):
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        mixtral_custom_decoderlayer_forward, layer
    )


def mixtral_custom_mlp_forward(self, hidden_states):
    current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
    self.feat = current_hidden_states.clone().detach().cpu().double()
    current_hidden_states = self.w2(current_hidden_states)
    return current_hidden_states
    

def enable_mixtral_custom_mlp(layer, layer_id):
    modified_modules = layer.block_sparse_moe.experts
    for modified_module in modified_modules:
        modified_module.layer_id = layer_id 
        modified_module.forward = types.MethodType(mixtral_custom_mlp_forward, modified_module)

    return modified_modules


def mixtral_custom_attention_forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
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

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones((q_len, q_len), device=hidden_states.device)).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
        causal_mask = causal_mask.expand(bsz, 1, q_len, q_len)
        attn_weights = attn_weights + causal_mask

        # ###################################################
        self.attn_logits = attn_weights.clone().detach().cpu().double()
        # ###################################################

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # ###################################################
        self.attn_probs = attn_weights.clone().detach().cpu().double()
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

def enable_mixtral_custom_attention(layer, layer_id):
    modified_module = layer.self_attn
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(mixtral_custom_attention_forward, modified_module)

    return modified_module
