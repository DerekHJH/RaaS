import logging
import os
from typing import List

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from benchmarks.evals.waterfall.main import MarkovConfigs

logger = logging.getLogger(__name__)

configs = MarkovConfigs(dataset="math500", model="Qwen/Qwen2.5-Math-7B-Instruct", approach="full")
tot_layer_ids = list(range(configs.model_config.num_hidden_layers))
tot_head_ids = list(range(configs.model_config.num_attention_heads))

important_layer_head_pairs = [
    (layer_id, head_id) for layer_id in tot_layer_ids for head_id in tot_head_ids
][:1]
important_layer_head_pairs = [(24, 2), (18, 8), (26, 12), (23, 18)]
if __name__ == "__main__":

    full_attentions: List[torch.Tensor] = torch.load(
        os.path.join(configs.result_path, "full_attentions.pt")
    )
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    for i, (layer_id, head_id) in enumerate(important_layer_head_pairs):

        ax = axs[i // 2, i % 2]

        logger.info(f"Processing layer {layer_id}, head {head_id}")

        attns = full_attentions[layer_id][0, head_id, ...].cpu().float()

        # Put all nozero numbers in attns into a list
        attns_list = []
        for i in range(attns.shape[0]):
            for j in range(attns.shape[1]):
                if attns[i, j] > 0:
                    attns_list.append(attns[i, j])

        # Step 1: Create bins and compute histogram
        bins = np.linspace(0, 1, num=100)  # Define your bins (10 bins in this case)
        hist, edges = np.histogram(attns_list, bins=bins)

        # Step 2: Plot the histogram as a bar plot
        hist[0] = hist[1]
        ax.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")

        ax.set_xlabel("Attention score")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-0.1, 1.1)

    plt.savefig(
        os.path.join("results", f"fig-algo-alpha.pdf"),
        dpi=300,
        bbox_inches="tight",
        format="pdf",
    )
    plt.close()
