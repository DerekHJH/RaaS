import logging
import os
from typing import Tuple

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from benchmarks.evals.markov.main import MarkovConfigs

logger = logging.getLogger(__name__)

configs = MarkovConfigs(dataset="math500", model="peiyi9979/mistral-7b-sft", approach="full")
cache_budget = 128
page_size = 16

important_layer_head_pairs = [
    (0, 2),
    (0, 3),
    (0, 10),
    (0, 16),
    (0, 26),
    (0, 29),
    (0, 30),
    (0, 31),
    (1, 0),
    (1, 2),
    (1, 3),
    (1, 8),
    (1, 9),
    (1, 16),
    (1, 20),
    (1, 21),
    (1, 23),
    (2, 0),
    (2, 3),
    (2, 10),
    (2, 20),
    (2, 21),
    (5, 0),
    (5, 6),
    (5, 7),
    (5, 8),
    (5, 22),
    (6, 2),
    (6, 15),
    (7, 5),
    (7, 18),
    (7, 19),
    (9, 5),
    (30, 27),
    (31, 2),
]


def get_sink_attention(attention: torch.Tensor) -> torch.Tensor:
    """
    Get the simulated attention map when using the streamingllm algorithm.
    """
    attention = attention.clone()
    for i in range(cache_budget, attention.shape[0]):
        attention[i, 4 : -cache_budget + 4] = 0
        row_sum = attention[i].sum()
        attention[i] /= row_sum
    return attention


def get_h2o_attention(attention: torch.Tensor) -> torch.Tensor:
    """
    Get the simulated attention map when using the h2o algorithm.
    """
    attention = attention.clone()  # (seq_len, seq_len)
    cum_attn_score = torch.cumsum(attention, dim=0)  # (seq_len, seq_len)
    evict_ids = []

    for i in range(cache_budget, attention.shape[0]):
        # Find the token with the smallest cumulative attention score, which is to be evicted
        evict_id = torch.argmin(cum_attn_score[i, : i + 1 - cache_budget // 2]).item()
        assert evict_id not in evict_ids, f"evict_id {evict_id} already in evict_ids {evict_ids}"
        evict_ids.append(evict_id)

        # Mark the token to be evicted
        cum_attn_score[:, evict_id] = 2  # Magic number as long as it is greater than 1
        attention[i, cum_attn_score[i, :] == 2] = 0
        assert (
            attention[i, :] != 0
        ).sum() == cache_budget, "Attention has more than cache_budget non-zero elements"

        row_sum = attention[i].sum()
        attention[i] /= row_sum
    return attention


def get_quest_attention(attention: torch.Tensor) -> torch.Tensor:
    """
    Get the simulated attention map when using the quest algorithm.
    """

    attention = attention.clone()
    k = cache_budget // page_size - 1  # Always choose the last page
    for i in range(cache_budget, attention.shape[0]):

        # We always keep the last page. So we only discuss whether to keep preceding pages
        line_attention = attention[i, : i // page_size * page_size]
        mask = torch.zeros_like(line_attention, dtype=torch.bool)
        line_attention = line_attention.reshape(-1, page_size)

        the_max_value_in_each_page = line_attention.max(dim=-1).values
        topk_page_ids = the_max_value_in_each_page.topk(k).indices

        token_ids_in_the_topk_pages = topk_page_ids.unsqueeze(-1).repeat(
            1, page_size
        ) * page_size + torch.arange(page_size, device=topk_page_ids.device)
        token_ids_in_the_topk_pages = token_ids_in_the_topk_pages.reshape(-1)

        mask.scatter_(0, token_ids_in_the_topk_pages, True)
        line_attention = line_attention.reshape(-1)
        line_attention[~mask] = 0

        row_sum = attention[i].sum()
        attention[i] /= row_sum
    return attention


def get_raas_attention(attention: torch.Tensor) -> torch.Tensor:
    """
    Get the simulated attention map when using the raas algorithm.
    """

    attention = attention.clone()
    # k = cache_budget // page_size - 1  # Always choose the last page
    # for i in range(cache_budget, attention.shape[0]):
    #     import pdb

    #     pdb.set_trace()
    #     # We always keep the last page. So we only discuss whether to keep preceding pages
    #     line_attention = attention[i, : i // page_size * page_size]
    #     mask = torch.zeros_like(line_attention, dtype=torch.bool)
    #     line_attention = line_attention.reshape(-1, page_size)

    #     the_max_value_in_each_page = line_attention.max(dim=-1).values
    #     topk_page_ids = the_max_value_in_each_page.topk(k).indices

    #     token_ids_in_the_topk_pages = topk_page_ids.unsqueeze(-1).repeat(
    #         1, page_size
    #     ) * page_size + torch.arange(page_size, device=topk_page_ids.device)
    #     token_ids_in_the_topk_pages = token_ids_in_the_topk_pages.reshape(-1)

    #     mask.scatter_(0, token_ids_in_the_topk_pages, True)
    #     line_attention = line_attention.reshape(-1)
    #     line_attention[~mask] = 0

    #     row_sum = attention[i].sum()
    #     attention[i] /= row_sum
    return attention


def square_attention(attentions: torch.Tensor, layer_id, head_id) -> torch.Tensor:
    """
    Arrange the attention map into a square matrix.

    Args:
        attentions: Tuple (of length `seq_len`) of Tuple (of length `num_layers`) of
        torch.Tensor --- `seq_len` * `num_layers` torch.Tensor in total,
        each of shape (`batch_size`, `num_heads`, `num_attend_tokens`, `num_attended_tokens`).

        assert len(attentions) == seq_len
        assert len(attentions[0]) == num_layers
        assert attentions[0][0].shape == (batch_size, num_heads, num_prefill_tokens, num_prefill_tokens)
        assert attentions[1][0].shape == (batch_size, num_heads, 1, num_prefill_tokens + 1)
        assert attentions[2][0].shape == (batch_size, num_heads, 1, num_prefill_tokens + 2)

        layer_id: int, the layer index of the attention map to be plotted.
        head_id: int, the head index of the attention map to be plotted.

    Returns:
        torch.Tensor, the arranged attention map, with shape (`seq_len`, `seq_len`).
    """
    # A list (of length seq_len) torch.Tensor,
    # each with shape (num_attend_tokens, num_attended_tokens)
    attention = [attentions[i][layer_id][0, head_id, :, :] for i in range(len(attentions))]
    """
    assert attention[0].shape == (num_prefill_tokens, num_prefill_tokens)
    assert attention[1].shape == (1, num_prefill_tokens + 1)
    assert attention[2].shape == (1, num_prefill_tokens + 2)
    """
    assert attention[0].shape[0] == attention[0].shape[1]
    assert attention[1].shape == (1, attention[0].shape[1] + 1)
    assert attention[2].shape == (1, attention[0].shape[1] + 2)

    for i, tensor in enumerate(attention):
        padding = (0, attention[-1].shape[1] - tensor.shape[1])  # (left, right)
        attention[i] = torch.nn.functional.pad(tensor, padding, mode="constant", value=0)

    attention = torch.cat(attention, dim=0).cpu().float()  # shape (seq_len, seq_len)
    return attention


if __name__ == "__main__":

    attentions: Tuple[Tuple[torch.Tensor]] = torch.load(
        os.path.join(configs.result_path, "attentions.pt")
    )

    for layer_id in configs.layer_ids:
        for head_id in configs.head_ids:

            logger.info(f"Plotting attention map for layer {layer_id}, head {head_id}")

            attention = square_attention(attentions, layer_id, head_id)

            # Scale the attention score For better visibility

            # Plot
            fig, axs = plt.subplots(1, 5, figsize=(20, 4))
            red_black_cmap = LinearSegmentedColormap.from_list("RedBlack", ["black", "red"])

            sns.heatmap(attention, cmap=red_black_cmap, ax=axs[0], cbar=False)
            sns.heatmap(get_sink_attention(attention), cmap=red_black_cmap, ax=axs[1], cbar=False)
            sns.heatmap(get_h2o_attention(attention), cmap=red_black_cmap, ax=axs[2], cbar=False)
            sns.heatmap(get_quest_attention(attention), cmap=red_black_cmap, ax=axs[3], cbar=False)
            sns.heatmap(get_raas_attention(attention), cmap=red_black_cmap, ax=axs[4], cbar=False)

            axs[0].set_title("full")
            axs[1].set_title("sink")
            axs[2].set_title("h2o")
            axs[3].set_title("quest")
            axs[4].set_title("raas")

            plt.savefig(
                os.path.join(configs.result_path, f"layer_{layer_id}_head_{head_id}.png"),
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()
