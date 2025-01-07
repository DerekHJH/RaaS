import logging
import os
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

from benchmarks.evals.markov.main import MarkovConfigs
from benchmarks.evals.markov.plot_attention_map import square_attention

logger = logging.getLogger(__name__)

configs = MarkovConfigs(dataset="math500", model="Qwen/Qwen2.5-Math-7B-Instruct", approach="full")
configs.layer_ids = list(range(configs.model_config.num_hidden_layers))
configs.head_ids = list(range(configs.model_config.num_attention_heads))
cache_budget = 128
page_size = 16

# important (layer_id, head_id):
# backslash (7, 18), hard to solve
# Waterfall (6, 15), (30, 27), (9, 5)
# configs.layer_ids = [7, 6]
# configs.head_ids = [18, 15]


if __name__ == "__main__":

    attentions: Tuple[Tuple[torch.Tensor]] = torch.load(
        os.path.join(configs.result_path, "attentions.pt")
    )

    fig, axs = plt.subplots(
        len(configs.layer_ids),
        len(configs.head_ids),
        figsize=(5 * len(configs.head_ids), 4 * len(configs.layer_ids)),
    )

    for x, layer_id in enumerate(configs.layer_ids):
        for y, head_id in enumerate(configs.head_ids):
            ax = axs[x, y]
            data = []
            logger.info(f"Processing layer {layer_id}, head {head_id}")

            attention = square_attention(attentions, layer_id, head_id)

            for i in range(attention.shape[0]):
                row = attention[i].clone()
                sorted_row, _ = torch.sort(row, descending=True)
                cumulative_sum = torch.cumsum(sorted_row, dim=0)
                # If one element is already larger than 0.99, we fill in 1.
                data.append(max(1, (cumulative_sum <= 0.99).sum().item()))

            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cdf, linestyle="-", linewidth=2)
            ax.set_title(f"Layer {layer_id}, Head {head_id}")
            ax.set_xlim(0, attention.shape[0])
            ax.set_ylim(0, 1)

    plt.savefig(
        os.path.join(configs.result_path, f"attention_sparsity"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
