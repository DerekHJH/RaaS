import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.evals.utils import (
    dataset_metrics_map,
    dataset_names_map,
    model_names_map,
)

all_datasets = ["gsm8k", "aime", "math500"]
all_models = ["peiyi9979/mistral-7b-sft", "Qwen/Qwen2.5-Math-7B-Instruct"]
all_approaches = [
    "sink-64",
    "sink-128",
    "sink-256",
    "sink-512",
    "sink-1024",
    "quest-64",
    "quest-128",
    "quest-256",
    "quest-512",
    "quest-1024",
    "full",
]
colors = (
    ["#FFCC99", "#FFB366", "#FF9933", "#FF8000", "#CC6600"]
    + ["#A3C1E0", "#7FB3D5", "#4DA6D6", "#2A9BD5", "#0076A8"]
    # + ["#FF3333"]
    + ["#33CC33"]
)
markers = ["s"] * 5 + ["*"] * 5 + ["o"] * 1

x_metric_name = "num_decode"  # ["num_decode", "TTFT", "JCT", "TPOT"]
y_metric_name = "accuracy"
max_y_metric = 0
max_x_metric = 0


if __name__ == "__main__":

    fig, axs = plt.subplots(
        len(all_datasets), len(all_models), figsize=(6 * len(all_models), 5 * len(all_datasets))
    )

    for i, dataset_name in enumerate(all_datasets):
        for j, model_name in enumerate(all_models):
            last_model_name = model_name.split("/")[-1]
            path = f"results/{dataset_name}/{last_model_name}/data.json"
            dataset = pd.read_json(path)

            for k, approach_name in enumerate(all_approaches):
                if f"accuracy_{approach_name}" not in dataset.columns:
                    continue
                # Could change the metric here
                x_metric = np.mean(dataset[f"{x_metric_name}_{approach_name}"])
                y_metric = np.mean(dataset[f"{y_metric_name}_{approach_name}"])
                max_y_metric = max(max_y_metric, y_metric)
                max_x_metric = max(max_x_metric, x_metric)
                axs[i][j].scatter(
                    x_metric,
                    y_metric,
                    label=approach_name,
                    s=100,
                    c=colors[k],
                    marker=markers[k],
                )
    # After max_x_metric and max_y_metric are determined
    max_x_metric *= 1.05
    max_y_metric *= 1.05
    for i, dataset_name in enumerate(all_datasets):
        for j, model_name in enumerate(all_models):
            axs[i][j].set_xlim(left=0, right=max_x_metric)
            axs[i][j].set_ylim(bottom=0, top=max_y_metric)
            axs[i][j].tick_params(axis="x", labelsize=14)
            axs[i][j].tick_params(axis="y", labelsize=14)

    axs[0][0].legend(loc="upper center", bbox_to_anchor=(1, 1.3), ncol=6, fontsize=12)

    # Draw model names
    for j, model_name in enumerate(all_models):
        axs[0][j].set_title(model_names_map[model_name], fontsize=20)

    # Draw x metric name
    for j in range(1, len(all_models) + 1):
        axs[-1][-j].set_xlabel("# decode tokens", fontsize=20)

    for i, dataset_name in enumerate(all_datasets):
        axs[i][0].set_ylabel(dataset_metrics_map[dataset_name], fontsize=20)
        axs[i][0].text(
            -0.3,
            0.5,
            dataset_names_map[dataset_name],
            fontsize=20,
            rotation="vertical",
            ha="left",
            va="center",
            transform=axs[i][0].transAxes,
        )

    # Save
    plt.savefig("results/acc_vs_speed.pdf", format="pdf", bbox_inches="tight", dpi=600)
