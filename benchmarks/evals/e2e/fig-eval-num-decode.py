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

dataset_name = "math500"
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
# all_models = [
#     # "peiyi9979/mistral-7b-sft",
#     "Qwen/Qwen2.5-Math-7B-Instruct",
#     # "AIDC-AI/Marco-o1",
#     # "agentica-org/DeepScaleR-1.5B-Preview",
# ]
all_approaches = [
    "sink-128",
    "h2o-128",
    "quest-128",
    "raas-128",
    "full",
]


if __name__ == "__main__":

    fig, axs = plt.subplots(1, 1, figsize=(6, 5))

    last_model_name = model_name.split("/")[-1]
    path = f"results/{dataset_name}/{last_model_name}/data.json"
    dataset = pd.read_json(path)

    for approach_name in all_approaches:
        decode_lengths = dataset[f"num_decode_{approach_name}"].tolist()
        sorted_decode_lengths = sorted(decode_lengths)
        cdf = np.arange(1, len(sorted_decode_lengths) + 1) / len(sorted_decode_lengths)
        axs.plot(
            sorted_decode_lengths,
            cdf,
            label=approach_name_map[approach_name.split("-")[0]] + "-128",
        )

    axs.set_xlabel("# decode tokens", fontsize=20)
    axs.set_ylabel("CDF", fontsize=20)

    axs.legend(fontsize=20)
    axs.tick_params(axis="x", labelsize=20)
    axs.tick_params(axis="y", labelsize=20)
    axs.set_xlim(0)
    axs.set_ylim(0, 1)

    # Save
    plt.savefig("results/fig-eval-num-decode.png", format="png", bbox_inches="tight", dpi=400)
