import logging
import math
import types
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Cache,
    Qwen2Attention,
    apply_rotary_pos_emb,
    repeat_kv,
)

logger = logging.getLogger(__name__)


def local_heavy_hitter_mask(attn_weights, cache_budget, page_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by page_size
    seq_length = attn_weights.shape[-1]
    padding_length = page_size - ((seq_length - 1) % page_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into page_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // page_size,
        page_size,
    ).amax(dim=-1)

    topk_score, topk = chunk_attn_weights.topk(
        k=min(max(3, cache_budget // page_size), chunk_attn_weights.size(-1)), dim=-1
    )
    access_page_ids = topk.clone()
    access_page_scores = topk_score.clone()

    # repeat topk page_size times and recover the original indexes (* page_size + arange(page_size))
    topk = topk.unsqueeze(-1).repeat(1, 1, 1, 1, page_size) * page_size + torch.arange(
        page_size, device=topk.device
    )
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom, access_page_ids, access_page_scores


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_idx < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,
        )

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    ############################
    # Start of Quest Attention #
    ############################
    sign = (query_states > 0) + (~(query_states > 0)) * -1
    max_key = key_states * sign
    postive_query = query_states * sign

    # expend max_key to be divisible by page_size
    seq_length = max_key.shape[-2]
    padding_length = self.page_size - ((seq_length - 1) % self.page_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into page_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.page_size,
        self.page_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key page_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.page_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        postive_query.float(),
        chunk_max_key.transpose(2, 3),
    )

    kv_seq_len = past_key_value.get_seq_length()

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):  # Do not support TP
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    # We do not accept external attention mask for RaaS Attention, we prepare the mask_bottom here
    assert attention_mask is None, "External attention mask is not supported for RaaS Attention"
    attention_mask = past_key_value.get_attention_mask(attn_weights, self.layer_idx)

    if attention_mask is not None:
        # The following assertion is to make sure all heads share the same attention mask
        # But quest and raas allow different attention masks for different heads
        # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
        #     )
        # Therefore, we remove the preceding assertion and add the following assertion
        # import pdb

        # pdb.set_trace()
        assert (
            attention_mask.shape == attn_weights.shape
        ), "Attention mask should have the same shape as attn_weights"
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        quantized_weight = quantized_weight + attention_mask
        quantized_weight = torch.max(
            quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
        )

    cache_budget = min(kv_seq_len, self.cache_budget)

    attn_weights_for_selection = quantized_weight

    if cache_budget > 0:
        mask_bottom, access_page_ids, access_page_scores = local_heavy_hitter_mask(
            attn_weights_for_selection, cache_budget, self.page_size
        )  # Default: No padding applied to input
        past_key_value.update_access_history(
            access_page_ids, access_page_scores, self.layer_idx
        )  # hjh
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)
    mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )
    ##########################
    # End of Quest Attention #
    ##########################

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def enable_raas_attention_eval(model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_raas_attention_eval(
                module,
                args,
            )

        if isinstance(module, (Qwen2Attention,)):
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(forward, model._modules[name])
            model._modules[name].cache_budget = args["cache_budget"]
            model._modules[name].page_size = args["page_size"]
