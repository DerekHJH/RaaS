import random

import numpy as np
import torch

from benchmarks.data_sets import math500, needle


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


str2class = {
    "math500": math500.Math500,
    "needle": needle.Needle,
}


model_names_map = {
    "peiyi9979/mistral-7b-sft": "Mistral 7B SFT",
}
dataset_names_map = {"math500": "MATH500", "needle": "Needle"}
dataset_metrics_map = {"math500": "F1 score", "needle": "F1 score"}
