import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from benchmarks.eval_engines.plot_utils import (
    dataset_metrics_map,
    dataset_names_map,
    model_names_map,
)

all_datasets = ["math500", "aime"]
all_models = ["peiyi9979/mistral-7b-sft", "Qwen/Qwen2.5-Math-7B-Instruct"]
all_approaches = ["full", "streamingllm"]
colors = (
    ["#33CC33"]
    + ["#FF3333"]
    + ["#CC6600", "#FF8000", "#FF9933", "#FFB366", "#FFCC99"]
    + ["#0076A8", "#2A9BD5", "#4DA6D6", "#7FB3D5", "#A3C1E0"]
)
markers = ["o"] * 2 + ["s"] * 5 + ["*"] * 5


if __name__ == "__main__":

    fig, axs = plt.subplots(
        len(all_datasets), len(all_models), figsize=(6 * len(all_models), 5 * len(all_datasets))
    )

    for i, dataset_name in enumerate(all_datasets):
        for j, model_name in enumerate(all_models):
            last_model_name = model_name.split("/")[-1]
            path = f"results/{dataset_name}/{last_model_name}/data.json"
            dataset = pd.read_json(path)
            # import pdb
            # pdb.set_trace()

            for k, approach_name in enumerate(all_approaches):
                accuracy_avg = np.mean(dataset[f"accuracy_{approach_name}"])
                TTFT_avg = np.mean(dataset[f"TTFT_{approach_name}"])
                JCT_avg = np.mean(dataset[f"JCT_{approach_name}"])
                TPOT_avg = np.mean(dataset[f"TPOT_{approach_name}"])
                num_decode_avg = np.mean(dataset[f"num_decode_{approach_name}"])
                axs[i][j].scatter(
                    TTFT_avg,
                    accuracy_avg,
                    label=approach_name,
                    s=100,
                    c=colors[k],
                    marker=markers[k],
                )

            axs[i][j].set_ylim(bottom=0)
            axs[i][j].set_xlim(left=0)
            axs[i][j].tick_params(axis="y", labelsize=14)
            axs[i][j].tick_params(axis="x", labelsize=14)

    axs[0][0].legend(loc="upper center", bbox_to_anchor=(1.3, 1.3), ncol=6, fontsize=12)

    # Draw model names
    for j, model_name in enumerate(all_models):
        axs[0][j].set_title(model_names_map[model_name], fontsize=20)

    for j in range(1, len(all_models) + 1):
        axs[-1][-j].set_xlabel("JCT/s", fontsize=20)

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
    plt.savefig("results/results.pdf", format="pdf", bbox_inches="tight")
