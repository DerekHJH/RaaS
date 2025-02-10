import logging
import os
from typing import Tuple

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from benchmarks.evals.waterfall.main import MarkovConfigs
from benchmarks.evals.waterfall.plot_attention_map import square_attention

logger = logging.getLogger(__name__)

configs = MarkovConfigs(dataset="math500", model="peiyi9979/mistral-7b-sft", approach="full")
cache_budget = 128
page_size = 16


if __name__ == "__main__":

    attentions: Tuple[Tuple[torch.Tensor]] = torch.load(
        os.path.join(configs.result_path, "attentions.pt")
    )
    configs.head_ids = [0]
    fig, axs = plt.subplots(len(configs.layer_ids), 1, figsize=(20, 4 * len(configs.layer_ids)))

    for layer_id in configs.layer_ids:
        ax = axs[layer_id]
        for head_id in configs.head_ids:

            logger.info(f"Attention score distribution for layer {layer_id}, head {head_id}")

            attention = square_attention(attentions, layer_id, head_id)

            # boxplot on ax
            sns.boxplot(
                data={
                    token_id: attention[token_id, : token_id + 1]
                    for token_id in range(0, attention.size(0), 10)
                },
                ax=ax,
            )

            # Plot

    plt.savefig(
        os.path.join(configs.result_path, f"attention_score_boxplot"),
        dpi=100,
        bbox_inches="tight",
    )
    plt.close()
