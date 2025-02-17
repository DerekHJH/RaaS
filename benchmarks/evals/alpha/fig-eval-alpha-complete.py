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

all_datasets = ["math500", "math500"]
all_models = ["peiyi9979/mistral-7b-sft", "Qwen/Qwen2.5-Math-7B-Instruct"]
all_approaches = [
    [
        "raas-64-0.005",
        "raas-128-0.005",
        "raas-256-0.005",
        "raas-512-0.005",
        "raas-1024-0.005",
    ],
    [
        "raas-64-0.01",
        "raas-128-0.01",
        "raas-256-0.01",
        "raas-512-0.01",
        "raas-1024-0.01",
    ],
    [
        "raas-64-0.02",
        "raas-128-0.02",
        "raas-256-0.02",
        "raas-512-0.02",
        "raas-1024-0.02",
    ],
    [
        "raas-64-0.05",
        "raas-128-0.05",
        "raas-256-0.05",
        "raas-512-0.05",
        "raas-1024-0.05",
    ],
]

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

            for k, approaches in enumerate(all_approaches):

                x_list = []
                y_list = []
                for approach_name in approaches:
                    if f"accuracy_{approach_name}" not in dataset.columns:
                        print(f"Skipping {approach_name}")
                        continue

                    x_metric = int(approach_name.split("-")[-2])
                    y_metric = np.mean(dataset[f"accuracy_{approach_name}"])
                    x_list.append(x_metric)
                    y_list.append(y_metric)

                if len(x_list) == 0 or len(y_list) == 0:
                    continue

                max_x_metric = max(max_x_metric, max(x_list))
                max_y_metric = max(max_y_metric, max(y_list))

                axs[i][j].plot(
                    x_list,
                    y_list,
                    label="alpha=" + approaches[0].split("-")[-1],
                    marker="*",
                    markersize=15,
                    linewidth=3,
                )

            # Plot full
            path = f"../e2e/results/{dataset_name}/{last_model_name}/data.json"
            dataset = pd.read_json(path)
            x_list = [0, max_x_metric]
            temp = np.mean(dataset[f"accuracy_full"])
            y_list = [temp, temp]
            axs[i][j].plot(
                x_list,
                y_list,
                label="Dense",
                linewidth=3,
                linestyle="--",
            )

            # Plot complex alpha
            path = f"../e2e/results/{dataset_name}/{last_model_name}/data.json"
            dataset = pd.read_json(path)

            x_list = []
            y_list = []
            for approach_name in ["raas-64", "raas-128", "raas-256", "raas-512", "raas-1024"]:
                if f"accuracy_{approach_name}" not in dataset.columns:
                    print(f"Skipping {approach_name}")
                    continue

                x_metric = int(approach_name.split("-")[-1])
                y_metric = np.mean(dataset[f"accuracy_{approach_name}"])
                x_list.append(x_metric)
                y_list.append(y_metric)

            axs[i][j].plot(
                x_list,
                y_list,
                label="alpha=*",
                marker="*",
                markersize=15,
                linewidth=3,
            )

            x_list = []
            y_list = []
            for approach_name in ["sink-128", "sink-256", "sink-512", "sink-1024"]:
                if f"accuracy_{approach_name}" not in dataset.columns:
                    print(f"Skipping {approach_name}")
                    continue

                x_metric = int(approach_name.split("-")[-1])
                y_metric = np.mean(dataset[f"accuracy_{approach_name}"])
                x_list.append(x_metric)
                y_list.append(y_metric)

            axs[i][j].plot(
                x_list,
                y_list,
                label="alpha=0",
                marker="*",
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

    axs[0][0].legend(loc="upper center", bbox_to_anchor=(1, 1.5), ncol=6, fontsize=20)

    # Draw model names
    for j, model_name in enumerate(all_models):
        axs[0][j].set_title(model_names_map[model_name], fontsize=20)

    # Draw x metric name
    for j in range(1, len(all_models) + 1):
        axs[-1][-j].set_xlabel("Cache budget / # tokens", fontsize=20)

    for i, dataset_name in enumerate(all_datasets):
        axs[i][0].set_ylabel(dataset_metrics_map[dataset_name], fontsize=20)
        axs[i][0].text(
            -0.4,
            0.5,
            dataset_names_map[dataset_name],
            fontsize=20,
            rotation="vertical",
            ha="left",
            va="center",
            transform=axs[i][0].transAxes,
        )

    # Save
    plt.savefig("results/fig-eval-alpha.pdf", format="pdf", bbox_inches="tight", dpi=400)
