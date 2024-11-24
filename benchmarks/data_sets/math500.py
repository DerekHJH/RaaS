import logging
import pandas as pd
import os
import numpy as np
from typing import Dict, List
import glob
from transformers import AutoTokenizer
from tqdm import tqdm
from functools import lru_cache
from benchmarks.data_sets.data_set import Data_set

logger = logging.getLogger(__name__)


class Math500(Data_set):

    def load_raw_data(self) -> pd.DataFrame:
        """
        Construct the dataset from a jsonl file
        """
        self.raw_data_file_path = 'raw_data/MATH500.jsonl'
        data = pd.read_json(self.raw_data_file_path, lines=True)
        return data

    def _create_answer_field(self, row: Dict) -> Dict:
        return row
    def _create_prompt_field(self, row: Dict) -> Dict:
        row['prompt'] = row['problem'] + '\nLet\'s think step by step:'
        return row


if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
    dataset = Math500(tokenizer=tokenizer)
    import pdb; pdb.set_trace()