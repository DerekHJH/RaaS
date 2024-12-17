import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":

    # Give me two axs
    fig, ax = plt.subplots(figsize=(8, 5), dpi=600)

    dataset = pd.read_json("results/long_inference.json")

    x_vals = np.sort(dataset["prefill_lengths"] // 1024)  # in K (unit)
    y_vals_1 = dataset["prefill_time"]
    y_vals_2 = dataset["decode_time"]

    bar1 = ax.bar(x_vals, y_vals_1, label="Prefill time")
    bar2 = ax.bar(x_vals, y_vals_2, bottom=y_vals_1, label="Decode time")  # Stack C on top of B

    ax.set_xlabel("# prefill tokens (# decode tokens = 32K - # prefill tokens)")
    ax.set_ylabel("time/s")
    ax.legend(fontsize=8)

    # Figure configurations
    plt.savefig(
        "results/prefill_decode_lengths_cdf.pdf", format="pdf", bbox_inches="tight", dpi=400
    )
