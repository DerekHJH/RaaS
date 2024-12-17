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
model_name = "AIDC-AI/Marco-o1"
approach_name = "full"
tokenizer = AutoTokenizer.from_pretrained(model_name)
longben_dataset_names = ["2wikimqa", "multi_news", "samsum", "passage_count", "lcc"]

if __name__ == "__main__":

    # Give me two axs
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.5), dpi=600)

    # prob all jsonl files in the directory
    # for file in glob.glob("/data0/hujunhao/data/longbench/*.jsonl"):
    for dataset_name in longben_dataset_names:
        file = f"/data0/hujunhao/data/longbench/{dataset_name}.jsonl"
        dataset = pd.read_json(file, lines=True)

        decode_lengths = np.array(
            [len(tokenizer.encode(answer[0])) for answer in dataset["answers"]]
        )
        prefill_lengths = np.array(dataset["length"]) - decode_lengths

        # CDF of prefill lengths and decode lengths

        axs[0].hist(
            prefill_lengths,
            bins=200,
            density=True,
            histtype="step",
            cumulative=True,
            label=f"P {dataset_names_map[dataset_name]}",
            linestyle="--",
            linewidth=1.5,
        )
        axs[0].hist(
            decode_lengths,
            bins=200,
            density=True,
            histtype="step",
            cumulative=True,
            label=f"D {dataset_names_map[dataset_name]}",
            linestyle="-",
            linewidth=1.5,
        )

        # Figure configurations

    # Math dataset
    last_model_name = model_name.split("/")[-1]
    for dataset_name in dataset_names:
        path = f"results/{dataset_name}/{last_model_name}/data.json"
        dataset = pd.read_json(path)
        prompts = dataset["prompt"].tolist()
        prefill_lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
        decode_lengths = dataset["num_decode_full"].tolist()

        axs[1].hist(
            prefill_lengths,
            bins=200,
            density=True,
            histtype="step",
            cumulative=True,
            label=f"P {dataset_names_map[dataset_name]}",
            linestyle="--",
            linewidth=1.5,
        )
        axs[1].hist(
            decode_lengths,
            bins=200,
            density=True,
            histtype="step",
            cumulative=True,
            label=f"D {dataset_names_map[dataset_name]}",
            linestyle="-",
            linewidth=1.5,
        )

    axs[0].set_xlabel("# tokens", fontsize=10)
    axs[1].set_xlabel("# tokens", fontsize=10)
    axs[0].set_ylabel("CDF", fontsize=10)
    axs[0].set_title("(a) Long-prefill workloads", y=-0.35)
    axs[1].set_title("(b) Long-decode workloads", y=-0.35)

    for ax in axs:
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlim(0, 33000)
        ax.set_ylim(0, 1)

    # Figure configurations
    plt.savefig(
        "results/prefill_decode_lengths_cdf.pdf", format="pdf", bbox_inches="tight", dpi=400
    )
