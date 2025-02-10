import math
import types
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint
from transformers.models.falcon.modeling_falcon import (
    apply_rotary_pos_emb,
    dropout_add,
)


def falcon_custom_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    **kwargs,
):
    residual = hidden_states

    if self.config.new_decoder_architecture and self.config.num_ln_in_parallel_attn == 2:
        attention_layernorm_out = self.ln_attn(hidden_states)
        mlp_layernorm_out = self.ln_mlp(hidden_states)
    else:
        attention_layernorm_out = self.input_layernorm(hidden_states)

    # Self attention.
    attn_outputs = self.self_attention(
        attention_layernorm_out,
        layer_past=layer_past,
        attention_mask=attention_mask,
        position_ids=position_ids,
        alibi=alibi,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
    )

    attention_output = attn_outputs[0]

    if not self.config.new_decoder_architecture:
        if self.config.parallel_attn:
            mlp_layernorm_out = attention_layernorm_out
        else:
            residual = dropout_add(
                attention_output, residual, self.config.attention_dropout, training=self.training
            )
            mlp_layernorm_out = self.post_attention_layernorm(residual)

    if (
        self.config.new_decoder_architecture
        and self.config.parallel_attn
        and self.config.num_ln_in_parallel_attn == 1
    ):
        mlp_layernorm_out = attention_layernorm_out

    outputs = attn_outputs[1:]

    # MLP.
    mlp_output = self.mlp(mlp_layernorm_out)

    if self.config.new_decoder_architecture or self.config.parallel_attn:
        mlp_output += attention_output

    output = dropout_add(mlp_output, residual, self.config.hidden_dropout, training=self.training)

    if use_cache:
        outputs = (output,) + outputs
    else:
        outputs = (output,) + outputs[1:]
        
    self.feat = output.clone().detach().cpu().double()

    return outputs  # hidden_states, present, attentions

def enable_falcon_custom_decoderlayer(layer, layer_id):
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        falcon_custom_decoderlayer_forward, layer
    )


def falcon_custom_mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.act(self.dense_h_to_4h(x))
    self.feat = x.clone().detach().cpu().double()
    x = self.dense_4h_to_h(x)
    return x

def enable_falcon_custom_mlp(layer, layer_id):
    modified_module = layer.mlp
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(falcon_custom_mlp_forward, modified_module)

    return modified_module


def falcon_custom_attention_forward(
    self,
    hidden_states: torch.Tensor,
    alibi: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    head_mask: Optional[torch.Tensor] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
    num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads
    # 3 x [batch_size, seq_length, num_heads, head_dim]
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
    
    # ###################################################
    self.query_states = query_layer.detach().cpu().clone()
    self.key_states = key_layer.detach().cpu().clone()
    self.value_states = value_layer.detach().cpu().clone()
    # ###################################################

    batch_size, query_length, _, _ = query_layer.shape

    query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)

    kv_seq_len = key_layer.shape[-2]
    if layer_past is not None:
        kv_seq_len += layer_past[0].shape[-2]
    if alibi is None:
        cos, sin = self.rotary_emb(value_layer, seq_len=kv_seq_len)
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin, position_ids)

    if layer_past is not None:
        past_key, past_value = layer_past
        # concatenate along seq_length dimension:
        #  - key: [batch_size, self.num_heads, kv_length, head_dim]
        #  - value: [batch_size, self.num_heads, kv_length, head_dim]
        key_layer = torch.cat((past_key, key_layer), dim=-2)
        value_layer = torch.cat((past_value, value_layer), dim=-2)

    kv_length = key_layer.shape[-2]
    if use_cache:
        present = (key_layer, value_layer)
    else:
        present = None

    if self._use_sdpa and query_layer.device.type == "cuda" and attention_mask is not None:
        # For torch<=2.1.2, SDPA with memory-efficient backend is bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        query_layer = query_layer.contiguous()
        key_layer = key_layer.contiguous()
        value_layer = value_layer.contiguous()

    if alibi is None:
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores /= math.sqrt(self.head_dim)
        
        causal_mask = torch.tril(torch.ones((query_length, query_length), device=hidden_states.device)).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
        causal_mask = causal_mask.expand(batch_size, 1, query_length, query_length)
        attention_scores = attention_scores + causal_mask

        # ###################################################
        self.attn_logits = attention_scores.detach().cpu().clone()
        # ###################################################

        attention_scores = F.softmax(attention_scores, dim=-1, dtype=hidden_states.dtype)
        # It is unclear why neither dropout nor head_mask is applied here (while it is with alibi).
       
        # ###################################################
        self.attn_probs = attention_scores.clone().detach().cpu().double()
        # ###################################################
        
        attn_output = attention_scores @ value_layer

        attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

        attn_output = self.dense(attn_output)

        if output_attentions:
            return attn_output, present, attention_scores
        else:
            return attn_output, present

    else:
        if self._use_sdpa and not output_attentions and head_mask is None:
            # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
            # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
            is_causal = True if self.is_causal and attention_mask is None and query_length > 1 else False
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)

            attn_output = self.dense(attn_output)
        else:
            matmul_result = query_layer @ key_layer.transpose(-1, -2)

            # change view to [batch_size, num_heads, q_length, kv_length]
            attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = attention_scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float32)

            attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
            attention_logits *= self.inv_norm_factor
            attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
            # [batch_size, num_heads, q_length, kv_length]
            attention_probs = self.attention_dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # change view [batch_size, num_heads, q_length, kv_length]
            attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)

            # matmul: [batch_size * num_heads, q_length, head_dim]
            attn_output = (attention_probs_reshaped @ value_layer).flatten(0, 1)

            # change view [batch_size, q_length, num_heads * head_dim]
            attn_output = self._merge_heads(attn_output)

            attn_output = self.dense(attn_output)

        if output_attentions:
            return attn_output, present, attention_probs
        else:
            return attn_output, present

def enable_falcon_custom_attention(layer, layer_id):
    modified_module = layer.self_attention
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(falcon_custom_attention_forward, modified_module)

    return modified_module
