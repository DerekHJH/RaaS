import logging
from typing import Dict

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from benchmarks.data_sets.data_set import Data_set
from benchmarks.data_sets.utils import extract_answer, math_equal

logger = logging.getLogger(__name__)


class Math500(Data_set):

    def load_raw_data(self) -> pd.DataFrame:
        """
        Download the dataset from huggingface datasets.
        """
        return load_dataset("HuggingFaceH4/MATH-500", split="test").to_pandas()

    def create_groundtruth_field(self, row: Dict) -> Dict:
        row["groundtruth"] = extract_answer(row["solution"], "math")
        return row

    def create_prompt_field(self, row: Dict) -> Dict:
        # TODO(hjh): Create more complex prompts
        row["prompt"] = (
            row["problem"] + "\nMake sure the final answer is standalone and in latex format."
        )
        return row

    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:

        model_output: str = extract_answer(row[f"output_{approach}"], "math")
        groundtruth: str = row["groundtruth"]
        row[f"accuracy_{approach}"] = math_equal(model_output, groundtruth)
        return row


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
    dataset = Math500(tokenizer=tokenizer)

    import pdb

    pdb.set_trace()
