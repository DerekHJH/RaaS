import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":

    # Give me two axs
    fig, ax = plt.subplots(figsize=(6, 5), dpi=600)

    dataset = pd.read_json("results/long_inference.json")

    x_vals = np.array((2**16 - dataset["prefill_lengths"]) // 1024)  # decode lengths in k (unit)
    y_vals_1 = np.array(dataset["prefill_time"])
    y_vals_2 = np.array(dataset["decode_time"])

    bar1 = ax.bar(x_vals[::-1], y_vals_1[::-1], label="Prefill time", color="tab:blue")
    bar2 = ax.bar(
        x_vals[::-1], y_vals_2[::-1], bottom=y_vals_1[::-1], label="Decode time", color="tab:orange"
    )  # Stack C on top of B

    ax.set_xlabel("# decode tokens / k", fontsize=16)
    ax.set_ylabel("time/s", fontsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16)
    ax.set_title("(c) Prefill and decode time breakdown", fontsize=20, y=-0.24)

    # Figure configurations
    plt.savefig(
        "results/fig-background-pd-breakdown.pdf", format="pdf", bbox_inches="tight", dpi=400
    )
