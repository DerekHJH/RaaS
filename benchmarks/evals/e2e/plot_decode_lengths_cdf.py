import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from transformers import AutoTokenizer

from benchmarks.evals.utils import (
    dataset_metrics_map,
    dataset_names_map,
    model_names_map,
)

dataset_names = ["aime", "math500", "gsm8k"]
model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
approach_name = "full"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if __name__ == "__main__":

    fig, ax = plt.subplots(1, 1, figsize=(8, 2.5), dpi=600)

    # Math dataset
    last_model_name = model_name.split("/")[-1]
    for dataset_name in dataset_names:
        path = f"results/{dataset_name}/{last_model_name}/data.json"
        dataset = pd.read_json(path)
        decode_lengths = dataset["num_decode_full"].tolist()
        ax.hist(
            decode_lengths,
            bins=200,
            density=True,
            histtype="step",
            cumulative=True,
            label=f"{dataset_names_map[dataset_name]}",
            linestyle="-",
            linewidth=1.5,
        )

    ax.set_xlabel("# tokens", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title("# decode tokens", y=-0.35)

    ax.legend(fontsize=9)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlim(0, 33000)
    ax.set_ylim(0, 1)

    # Figure configurations
    plt.savefig("results/decode_lengths_cdf.pdf", format="pdf", bbox_inches="tight", dpi=400)
