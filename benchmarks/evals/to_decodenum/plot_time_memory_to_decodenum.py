import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from benchmarks.evals.utils import (
    dataset_metrics_map,
    dataset_names_map,
    model_names_map,
)

dataset = "gsm8k"
model = "peiyi9979/mistral-7b-sft"
all_approaches = [
    # "raas_optimized-64",
    # "raas_optimized-128",
    # "raas_optimized-256",
    "raas_optimized-512",
    "raas_optimized-1024",
    # "quest_optimized-64",
    # "quest_optimized-128",
    # "quest_optimized-256",
    # "quest_optimized-512",
    "quest_optimized-1024",
    "full_optimized",
]
def get_label(s: str):
    return s.replace("_optimized", "")

if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.7))
    plt.subplots_adjust(wspace=0.3)

    last_model_name = model.split("/")[-1]
    path = f"results/{dataset}/{last_model_name}/data.json"
    dataset = pd.read_json(path)
    
    # draw the time to decode_num

    for approach in all_approaches:
        xs = []
        ys = []
        x_key = f"JCT_decode_{approach}"
        if x_key not in dataset.columns:
            logger.warning(f"Key {x_key} not found in the dataset")
            continue
        JCT = dataset[x_key]
        dp = len(JCT)
        bonus = 2 if approach == "raas_optimized-1024" else 0 # to show the nearly overlapped lines more clearly
        for i in JCT[0].keys():
            xs.append(int(i) / 1024)
            ys.append(np.mean([JCT[j][i] for j in range(dp)]) + bonus)
        axs[0].plot(xs, ys, label=get_label(approach))

    axs[0].legend()
    axs[0].set_xlabel("# decode tokens / k")
    axs[0].set_ylabel("JCT / s")

    # draw the memory to decode_num

    for approach in all_approaches:
        xs = []
        ys = []
        bytes_per_token_key = f"bytes_per_token_{approach}"
        memory_token_key = f"memory_token_decode_{approach}"
        if bytes_per_token_key not in dataset.columns:
            logger.warning(f"Key {bytes_per_token_key} not found in the dataset")
            continue
        if memory_token_key not in dataset.columns:
            logger.warning(f"Key {memory_token_key} not found in the dataset")
            continue
        bytes_per_token = dataset[bytes_per_token_key]
        memory_token = dataset[memory_token_key]
        bonus = 0.01 if approach == "quest_optimized-1024" else 0
        for i in memory_token[0].keys():
            xs.append(int(i) / 1024)
            ys.append(np.mean([bytes_per_token * memory_token[j][i] for j in range(dp)]) / 1024 ** 3 + bonus)
        axs[1].plot(xs, ys, label=get_label(approach))
    
    axs[1].set_xlabel("# decode tokens / k")
    axs[1].set_ylabel("KV Cache / GB")
        



    # Save

    plt.savefig("results/fig-eval-time-memory.pdf", format="pdf", bbox_inches="tight")
