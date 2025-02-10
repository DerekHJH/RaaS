import logging
import os
from typing import Tuple

import distinctipy
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from benchmarks.evals.waterfall.main import MarkovConfigs
from benchmarks.evals.waterfall.plot_attention_map import square_attention

logger = logging.getLogger(__name__)
colors = distinctipy.get_colors(300)


configs = MarkovConfigs(dataset="math500", model="peiyi9979/mistral-7b-sft", approach="full")
cache_budget = 128
page_size = 16

# configs.layer_ids = [0, 1]
# configs.head_ids = [0, 1]


if __name__ == "__main__":

    attentions: Tuple[Tuple[torch.Tensor]] = torch.load(
        os.path.join(configs.result_path, "attentions.pt")
    )

    fig, axs = plt.subplots(
        len(configs.layer_ids),
        len(configs.head_ids),
        figsize=(7 * len(configs.head_ids), 4 * len(configs.layer_ids)),
    )

    for layer_id in configs.layer_ids:
        for head_id in configs.head_ids:
            ax = axs[layer_id, head_id]

            logger.info(f"Attention score cdf for layer {layer_id}, head {head_id}")

            attention = square_attention(attentions, layer_id, head_id)

            # the axis is the token id while the y axis is the cumsum of the attention score
            cumsum = attention.cumsum(dim=-1)
            for i in range(cumsum.shape[0]):
                ax.plot(cumsum[i], linewidth=1, color=colors[i])
            ax.set_title(f"Layer {layer_id}, Head {head_id}")

    plt.savefig(
        os.path.join(configs.result_path, f"attention_score_cumsum"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
