import logging
import os
from typing import Tuple

# import distinctipy
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from benchmarks.evals.markov.main import MarkovConfigs
from benchmarks.evals.markov.plot_attention_map import square_attention

logger = logging.getLogger(__name__)
# colors = distinctipy.get_colors(300)

from matplotlib.colors import LinearSegmentedColormap

# Define the seven main colors (red, orange, yellow, green, blue, indigo, violet)
main_colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#8B00FF"]


# Function to generate color variants
def generate_color_variants(color, n_variants=10):
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#FFFFFF", color])
    return [cmap(i / n_variants) for i in range(n_variants)]


# Generate color variants for each main color
colors = []
for color in main_colors:
    colors.extend(generate_color_variants(color))


configs = MarkovConfigs(dataset="math500", model="peiyi9979/mistral-7b-sft", approach="full")
cache_budget = 128
page_size = 16

# important (layer_id, head_id):
# backslash (7, 18), hard to solve
# Waterfall (6, 15), (30, 27), (9, 5)
configs.layer_ids = [7, 6]
configs.head_ids = [18, 15]


if __name__ == "__main__":

    attentions: Tuple[Tuple[torch.Tensor]] = torch.load(
        os.path.join(configs.result_path, "attentions.pt")
    )

    fig, axs = plt.subplots(
        len(configs.layer_ids),
        len(configs.head_ids),
        figsize=(7 * len(configs.head_ids), 4 * len(configs.layer_ids)),
    )

    for x, layer_id in enumerate(configs.layer_ids):
        for y, head_id in enumerate(configs.head_ids):
            ax = axs[x, y]

            logger.info(f"Attention score cdf for layer {layer_id}, head {head_id}")

            attention = square_attention(attentions, layer_id, head_id)

            for i in range(attention.shape[1]):
                ax.plot(attention[:, i], linewidth=1, color=colors[i % len(colors)])
            ax.set_title(f"Layer {layer_id}, Head {head_id}")

    plt.savefig(
        os.path.join(configs.result_path, f"attention_score_importance"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
