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


if __name__ == "__main__":

    fig, axs = plt.subplots(
        len(all_datasets), len(all_models), figsize=(6 * len(all_models), 5 * len(all_datasets))
    )

    for i, dataset_name in enumerate(all_datasets):
        for j, model_name in enumerate(all_models):
            last_model_name = model_name.split("/")[-1]
            path = f"results/{dataset_name}/{last_model_name}/data.json"
            dataset = pd.read_json(path)

            labels = []
            data = []
            for approach_name in all_approaches:
                if f"num_decode_{approach_name}" not in dataset.columns:
                    continue
                labels.append(approach_name)
                data.append(dataset[f"num_decode_{approach_name}"])
            axs[i][j].boxplot(
                data,
                positions=range(len(data)),
                tick_labels=labels,
            )
            axs[i][j].tick_params(axis="x", rotation=45)

    # Draw model names
    for j, model_name in enumerate(all_models):
        axs[0][j].set_title(model_names_map[model_name], fontsize=20)

    # Draw x metric name
    for j in range(1, len(all_models) + 1):
        axs[-1][-j].set_xlabel(f"# decode tokens", fontsize=20)

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
    plt.savefig("results/num_decode.pdf", format="pdf", bbox_inches="tight")
