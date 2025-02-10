import math
import types
import warnings
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint


def mpt_custom_decoderlayer_forward(
    self,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_cache: bool = False,
    output_attentions: bool = False,
):
    # hidden_states: [batch_size, seq_length, hidden_size]
    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.norm_1(hidden_states)

    residual = hidden_states

    # Self attention.
    attn_outputs, attn_weights, past_key_value = self.attn(
        layernorm_output,
        position_bias=position_bias,
        attention_mask=attention_mask,
        past_key_value=layer_past,
    )

    hidden_states = self.resid_attn_dropout(attn_outputs) + residual

    layernorm_output = self.norm_2(hidden_states)

    # Get residual
    residual = hidden_states

    # MLP.
    output = self.ffn(layernorm_output, residual)
    
    self.feat = hidden_states.clone().detach().cpu().double()
    
    outputs = (output,)

    if use_cache:
        outputs += (past_key_value,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs  # hidden_states, present, attentions


def enable_mpt_custom_decoderlayer(layer, layer_id):
    """
    replace the forward function of MptBlock with a custom forward function `mpt_custom_decoderlayer_forward`
    """
    layer.layer_id = layer_id
    layer.forward = types.MethodType(
        mpt_custom_decoderlayer_forward, layer
    )
    

def mpt_custom_mlp_forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    hidden_states = self.act(self.up_proj(hidden_states))
    self.feat = hidden_states.clone().detach().cpu().double()
    intermediate_output = self.down_proj(hidden_states)

    output = F.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)
    output = output + residual

    return output
    

def enable_mpt_custom_mlp(layer, layer_id):
    """
    replace the forward function of MptMLP with a custom forward function `mpt_custom_mlp_forward`
    """
    modified_module = layer.ffn
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(mpt_custom_mlp_forward, modified_module)

    return modified_module


def mpt_custom_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_bias: torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    batch_size, seq_length = hidden_states.shape[:2]

    mixed_qkv = self.Wqkv(hidden_states)
    if self.clip_qkv:
        mixed_qkv = mixed_qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

    query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
    
    ##################################################################
    self.query_states = query_states.detach().cpu().clone()
    self.key_states = key_states.detach().cpu().clone()
    self.value_states = value_states.detach().cpu().clone()
    # ###################################################
    
    query_states = query_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.reshape(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)

    if past_key_value is not None:
        if len(past_key_value) != 0:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states)
    else:
        past_key_value = (key_states, value_states)

    attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.softmax_scale

    query_length = seq_length if past_key_value is None else seq_length + past_key_value[0].shape[2]

    if position_bias is not None:
        if len(position_bias.shape) != 3:
            raise ValueError(f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}")
        key_length = key_states.shape[-2]

        position_bias_query_index = max(0, position_bias.size(1) - query_length)
        position_bias_key_index = max(0, position_bias.size(2) - key_length)

        position_bias = position_bias[:, position_bias_query_index:, position_bias_key_index:]

        attention_scores = attention_scores + position_bias

    causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=hidden_states.device)).unsqueeze(0).unsqueeze(0)
    causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
    causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
    attention_scores = attention_scores + causal_mask

    # ###################################################
    self.attn_logits = attention_scores.clone().detach().cpu().double()
    # ###################################################

    # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(value_states.dtype)
    
    # ###################################################
    self.attn_probs = attn_weights.clone().detach().cpu().double()
    # ###################################################
    
    attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout_p, training=self.training)

    context_states = torch.matmul(attn_weights, value_states)
    context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)
    attn_output = self.out_proj(context_states)

    return attn_output, attn_weights, past_key_value
    
    
def enable_mpt_custom_attention(layer, layer_id):
    """
    replace the forward function of MptAttention with a custom forward function `mpt_custom_attention_forward`
    """
    modified_module = layer.attn
    modified_module.layer_id = layer_id 
    modified_module.forward = types.MethodType(mpt_custom_attention_forward, modified_module)

    return modified_module