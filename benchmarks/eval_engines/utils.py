import random

import numpy as np
import torch

from benchmarks.data_sets import aime, math500


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


str2class = {
    "math500": math500.Math500,
    "aime": aime.AIME,
}
