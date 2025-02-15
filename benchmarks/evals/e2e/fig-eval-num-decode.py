import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.evals.utils import (
    approach_name_map,
    dataset_metrics_map,
    dataset_names_map,
    model_names_map,
)

all_datasets = ["gsm8k", "aime", "math500"]
all_models = [
    "peiyi9979/mistral-7b-sft",
    "Qwen/Qwen2.5-Math-7B-Instruct",
]
all_approaches = [
    [
        "sink-64",
        "sink-128",
        "sink-256",
        "sink-512",
        "sink-1024",
    ],
    [
        "h2o-64",
        "h2o-128",
        "h2o-256",
        "h2o-512",
        "h2o-1024",
    ],
    [
        "quest-64",
        "quest-128",
        "quest-256",
        "quest-512",
        "quest-1024",
    ],
    [
        "raas-64",
        "raas-128",
        "raas-256",
        "raas-512",
        "raas-1024",
    ],
    ["full"],
]

markers = ["s"] + ["o"] + ["^"] + ["*"]

max_y_metric = 1
max_x_metric = 1024


if __name__ == "__main__":

    fig, axs = plt.subplots(
        len(all_datasets), len(all_models), figsize=(6 * len(all_models), 5 * len(all_datasets))
    )

    for i, dataset_name in enumerate(all_datasets):
        for j, model_name in enumerate(all_models):
            last_model_name = model_name.split("/")[-1]
            path = f"results/{dataset_name}/{last_model_name}/data.json"
            dataset = pd.read_json(path)

            for k, approaches in enumerate(all_approaches):

                x_list = []
                y_list = []
                for approach_name in approaches:
                    if f"num_decode_{approach_name}" not in dataset.columns:
                        print(f"Skipping {approach_name}")
                        continue

                    if "full" in approach_name:
                        x_list = [0, max_x_metric]
                        temp = np.mean(dataset[f"num_decode_{approach_name}"])
                        y_list = [temp, temp]
                        break

                    x_metric = int(approach_name.split("-")[-1])
                    y_metric = np.mean(dataset[f"num_decode_{approach_name}"])
                    x_list.append(x_metric)
                    y_list.append(y_metric)

                if len(x_list) == 0 or len(y_list) == 0:
                    print(f"Skipping {approach_name}")
                    continue

                max_x_metric = max(max_x_metric, max(x_list))
                max_y_metric = max(max_y_metric, max(y_list))

                if "full" in approach_name:
                    axs[i][j].plot(
                        x_list,
                        y_list,
                        label=approach_name_map[approaches[0].split("-")[0]],
                        linewidth=3,
                        linestyle="--",
                    )
                else:
                    axs[i][j].plot(
                        x_list,
                        y_list,
                        label=approach_name_map[approaches[0].split("-")[0]],
                        marker=markers[k],
                        markersize=15,
                        linewidth=3,
                    )

    # After max_x_metric and max_y_metric are determined
    max_x_metric *= 1.05
    max_y_metric *= 1.05
    for i, dataset_name in enumerate(all_datasets):
        for j, model_name in enumerate(all_models):
            axs[i][j].set_xlim(left=0, right=max_x_metric)
            axs[i][j].set_ylim(bottom=0, top=max_y_metric)
            axs[i][j].tick_params(axis="x", labelsize=20)
            axs[i][j].tick_params(axis="y", labelsize=20)

    axs[0][0].legend(loc="upper center", bbox_to_anchor=(1, 1.3), ncol=6, fontsize=20)

    # Draw model names
    for j, model_name in enumerate(all_models):
        axs[0][j].set_title(model_names_map[model_name], fontsize=20)

    # Draw x metric name
    for j in range(1, len(all_models) + 1):
        axs[-1][-j].set_xlabel("Cache budget / # tokens", fontsize=20)

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
    plt.savefig("results/fig-eval-num-decode.pdf", format="pdf", bbox_inches="tight", dpi=400)
