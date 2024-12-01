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

    # def extract_ans_from_model_output(self, model_output):

    #     # Remove 'kn' from the mistral 7b model
    #     pred_str = pred_str.replace("\u043a\u0438", "")

    #     if "final answer is $" in pred_str and "$. I hope" in pred_str:
    #         # minerva_math
    #         tmp = pred_str.split("final answer is $", 1)[1]
    #         pred = tmp.split("$. I hope", 1)[0].strip()
    #     elif "boxed" in pred_str:
    #         ans = pred_str.split("boxed")[-1]
    #         if len(ans) == 0:
    #             return ""
    #         elif ans[0] == "{":
    #             stack = 1
    #             a = ""
    #             for c in ans[1:]:
    #                 if c == "{":
    #                     stack += 1
    #                     a += c
    #                 elif c == "}":
    #                     stack -= 1
    #                     if stack == 0:
    #                         break
    #                     a += c
    #                 else:
    #                     a += c
    #         else:
    #             a = ans.split("$")[0].strip()
    #         pred = a
    #     elif "he answer is" in pred_str:
    #         pred = pred_str.split("he answer is")[-1].strip()
    #     elif "final answer is" in pred_str:
    #         pred = pred_str.split("final answer is")[-1].strip()
    #     elif "答案是" in pred_str:
    #         # Handle Chinese few-shot multiple choice problem answer extraction
    #         pred = pred_str.split("答案是")[1].strip().split("\n\n")[0].strip()
    #     else:  # use the last number
    #         if use_last_number:
    #             pattern = "-?\d*\.?\d+"
    #             pred = re.findall(pattern, pred_str.replace(",", ""))
    #             if len(pred) >= 1:
    #                 pred = pred[-1]
    #             else:
    #                 pred = ""
    #         else:
    #             pred = ""

    #     # multiple line
    #     # pred = pred.split("\n")[0]
    #     pred = re.sub(r"\n\s*", "", pred)
    #     if pred != "" and pred[0] == ":":
    #         pred = pred[1:]
    #     if pred != "" and pred[-1] == ".":
    #         pred = pred[:-1]
    #     if pred != "" and pred[-1] == "/":
    #         pred = pred[:-1]
    #     pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    #     return pred


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("peiyi9979/mistral-7b-sft")
    dataset = Math500(tokenizer=tokenizer)
    import pdb

    pdb.set_trace()
