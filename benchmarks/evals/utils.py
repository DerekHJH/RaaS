import random

import numpy as np
import torch

from benchmarks.data_sets import aime, gsm8k, math500


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Utils for datasets
str2class = {
    "math500": math500.Math500,
    "aime": aime.AIME,
    "gsm8k": gsm8k.GSM8k,
}


# Utils for plottings
model_names_map = {
    "peiyi9979/mistral-7b-sft": "Mistral 7B SFT",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Qwen 2.5 Math 7B Instruct",
    "AIDC-AI/Marco-o1": "Marco o1",
    "agentica-org/DeepScaleR-1.5B-Preview": "DeepScaleR 1.5B Preview",
}
dataset_names_map = {
    "math500": "MATH500",
    "aime": "AIME",
    "gsm8k": "GSM8K",
    "2wikimqa": "2WikiMQA",
    "multi_news": "MultiNews",
    "samsum": "SAMSum",
    "passage_count": "PassageCount",
    "lcc": "LCC",
}
dataset_metrics_map = {
    "math500": "Accuracy",
    "aime": "Accuracy",
    "gsm8k": "Accuracy",
}


approach_name_map = {
    "full": "Dense",
    "sink": "Sink",
    "h2o": "H2O",
    "quest": "Quest",
    "raas": "RaaS",
}
