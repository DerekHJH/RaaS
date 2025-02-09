import logging
import os
from typing import Tuple

import torch
from matplotlib import pyplot as plt

from benchmarks.evals.waterfall.main import MarkovConfigs
from benchmarks.evals.waterfall.plot_attention_map import square_attention

logger = logging.getLogger(__name__)


configs = MarkovConfigs(dataset="math500", model="Qwen/Qwen2.5-Math-7B-Instruct", approach="full")
tot_layer_ids = list(range(configs.model_config.num_hidden_layers))
tot_head_ids = list(range(configs.model_config.num_attention_heads))
cache_budgets = [64, 128, 256, 512, 1024]
threshold = 0.01
page_size = 16

# important (layer_id, head_id):
# backslash (7, 18), hard to solve
# Waterfall (6, 15), (30, 27), (9, 5)
tot_layer_ids = [0, 27]
tot_head_ids = [0, 1, 2, 25, 26, 27]


if __name__ == "__main__":

    attentions: Tuple[Tuple[torch.Tensor]] = torch.load(
        os.path.join(configs.result_path, "attentions.pt")
    )

    fig, axs = plt.subplots(
        len(tot_layer_ids),
        len(tot_head_ids),
        figsize=(7 * len(tot_head_ids), 4 * len(tot_layer_ids)),
    )

    for x, layer_id in enumerate(tot_layer_ids):
        for y, head_id in enumerate(tot_head_ids):
            ax = axs[x, y]

            logger.info(f"Processing layer {layer_id}, head {head_id}")

            attention = square_attention(attentions, layer_id, head_id)
            data = []
            for cache_budget in cache_budgets:
                num_phoenix = 0
                for j in range(attention.shape[1]):  # each columne
                    max_length = 0
                    cur_length = 0
                    for i in range(j, attention.shape[0]):  # each row
                        if attention[i, j] < threshold:
                            cur_length += 1
                        else:
                            max_length = max(
                                max_length, cur_length
                            )  # how long the phoenix is dead before reborn
                            cur_length = 0
                    if (
                        max_length > cache_budget
                    ):  # This phoenix is dead for too long and reborn to shock the cache
                        num_phoenix += 1
                data.append(num_phoenix)

            ax.plot(cache_budgets, data, linestyle="-", linewidth=2)
            ax.set_title(f"Layer {layer_id}, Head {head_id}")
            ax.set_xlabel("Cache Budget / tokens")
            ax.set_ylabel("# Phoenix")

    plt.savefig(
        os.path.join(configs.result_path, f"phoenix"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
