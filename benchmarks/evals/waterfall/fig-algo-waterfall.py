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

important_layer_head_pairs = [(24, 2), (18, 8), (26, 12), (23, 18)]


if __name__ == "__main__":

    full_attentions: List[torch.Tensor] = torch.load(
        os.path.join(configs.result_path, "full_attentions.pt")
    )
    import pdb

    pdb.set_trace()

    for layer_id, head_id in important_layer_head_pairs:

        logger.info(f"Processing layer {layer_id}, head {head_id}")

        # Scale the attention score For better visibility

        # Plot
        fig, axs = plt.subplots(1, 1, figsize=(5, 4))

        sns.heatmap(
            full_attentions[layer_id][0, head_id, ...].cpu().float(),
            cmap="viridis",
            ax=axs,
            cbar=False,
        )
        axs.set_xticks([])  # Remove x-axis ticks
        axs.set_yticks([])  # Remove y-axis ticks

        plt.savefig(
            os.path.join("results", f"layer_{layer_id}_head_{head_id}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
