import logging
import os
from typing import Dict

import pandas as pd
from transformers import AutoTokenizer

from benchmarks.data_sets.data_set import Data_set
from benchmarks.data_sets.utils import qa_f1_score

logger = logging.getLogger(__name__)


class Math500(Data_set):

    def load_raw_data(self) -> pd.DataFrame:
        """
        Construct the dataset from a jsonl file
        """
        raw_data_file_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "raw_data/MATH500.jsonl",
        )
        data = pd.read_json(raw_data_file_path, lines=True)
        return data

    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:
        # TODO: Replace qa_f1_score with an appropriate metric
        row[f"accuracy_{approach}"] = qa_f1_score(row[f"output_{approach}"], row["answer"])
        return row

    def _create_answer_field(self, row: Dict) -> Dict:
        return row

    def _create_prompt_field(self, row: Dict) -> Dict:
        row["prompt"] = row["problem"] + "\nLet's think step by step:"
        return row


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
    dataset = Math500(tokenizer=tokenizer)
    import pdb

    pdb.set_trace()
