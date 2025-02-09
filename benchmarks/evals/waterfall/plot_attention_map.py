import logging
import os
from typing import List

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from benchmarks.evals.waterfall.main import MarkovConfigs

logger = logging.getLogger(__name__)

configs = MarkovConfigs(dataset="math500", model="Qwen/Qwen2.5-Math-7B-Instruct", approach="full")
tot_layer_ids = list(range(configs.model_config.num_hidden_layers))
tot_head_ids = list(range(configs.model_config.num_attention_heads))
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


if __name__ == "__main__":

    full_attentions: List[torch.Tensor] = torch.load(
        os.path.join(configs.result_path, "full_attentions.pt")
    )
    # sink_attentions: List[torch.Tensor] = torch.load(
    #     os.path.join(configs.result_path, "sink-128_attentions.pt")
    # )
    h2o_attentions: List[torch.Tensor] = torch.load(
        os.path.join(configs.result_path, "h2o-512_attentions.pt")
    )
    quest_attentions: List[torch.Tensor] = torch.load(
        os.path.join(configs.result_path, "quest-128_attentions.pt")
    )
    raas_attentions: List[torch.Tensor] = torch.load(
        os.path.join(configs.result_path, "raas-128_attentions.pt")
    )

    for layer_id in tot_layer_ids:
        for head_id in tot_head_ids:

            logger.info(f"Processing layer {layer_id}, head {head_id}")

            # Scale the attention score For better visibility

            # Plot
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))

            sns.heatmap(
                full_attentions[layer_id][0, head_id, ...].cpu().float(),
                cmap="viridis",
                ax=axs[0],
                cbar=False,
            )
            sns.heatmap(
                h2o_attentions[layer_id][0, head_id, ...].cpu().float(),
                cmap="viridis",
                ax=axs[1],
                cbar=False,
            )
            sns.heatmap(
                quest_attentions[layer_id][0, head_id, ...].cpu().float(),
                cmap="viridis",
                ax=axs[2],
                cbar=False,
            )
            sns.heatmap(
                raas_attentions[layer_id][0, head_id, ...].cpu().float(),
                cmap="viridis",
                ax=axs[3],
                cbar=False,
            )

            axs[0].set_title("full")
            axs[1].set_title("h2o")
            axs[2].set_title("quest")
            axs[3].set_title("raas")

            plt.savefig(
                os.path.join(configs.result_path, f"layer_{layer_id}_head_{head_id}.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close()
