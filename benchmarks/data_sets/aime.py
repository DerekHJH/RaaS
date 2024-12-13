import logging
from typing import Dict

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from benchmarks.data_sets.data_set import Data_set
from benchmarks.data_sets.utils import extract_answer, math_equal, rouge_score

logger = logging.getLogger(__name__)


class AIME(Data_set):

    def load_raw_data(self) -> pd.DataFrame:
        """
        Download the dataset from huggingface datasets.
        """
        return load_dataset("qq8933/AIME_1983_2024", split="train").to_pandas()

    def create_groundtruth_field(self, row: Dict) -> Dict:
        row["groundtruth"] = str(
            row["Answer"]
        )  # No need to extract the answer becuase it is already in good format
        return row

    def create_prompt_field(self, row: Dict) -> Dict:
        # TODO(hjh): Create more complex prompts
        row["prompt"] = (
            row["Question"] + "\nMake sure the final answer is standalone and in latex format."
        )
        return row

    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:

        model_output: str = extract_answer(row[f"output_{approach}"], "aime")
        groundtruth: str = row["groundtruth"]
        import pdb

        pdb.set_trace()
        row[f"accuracy_{approach}"] = math_equal(model_output, groundtruth)
        row[f"final_output_{approach}"] = model_output
        # row[f"accuracy_{approach}"] = rouge_score(model_output, groundtruth)
        return row


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
    dataset = AIME(tokenizer=tokenizer)

    import pdb

    pdb.set_trace()
