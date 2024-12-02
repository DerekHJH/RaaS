import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List

# import torch
from tqdm.contrib import tenumerate
from transformers import Pipeline

from benchmarks.data_sets.data_set import Data_set
from benchmarks.eval_engines.eval_engine import Configs, EvalEngine

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class MarkovConfigs(Configs):
    # Overriding the default values of the parent class.
    tot_num_data: int = 1
    all_datasets: List[str] = field(default_factory=lambda: ["math500"])  # Fixed mutable default
    all_models: List[str] = field(default_factory=lambda: ["peiyi9979/mistral-7b-sft"])
    all_approaches: List[str] = field(default_factory=lambda: ["full"])


class MarkovEvalEngine(EvalEngine):

    def run_inference(self, pipe: Pipeline, dataset: Data_set) -> Data_set:
        """
        Run the inference and record the results into the dataset.
        """
        results = defaultdict(list)
        for i, (prompt, answer) in tenumerate(dataset, desc="dataset", leave=False):
            model_output, attentions = self.test_model(pipe, prompt, answer)
            # results[f"output_{self.configs.approach}"].append(model_output)
            results[f"output_{self.configs.approach}"].append(model_output)
        dataset.update(results)
        dataset.save_dataset(self.configs.result_path)

        return dataset

    def test_model(self, pipe, prompt, answer) -> Any:

        input_ids = pipe.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        model_output = pipe.model.generate(
            input_ids,
            max_length=pipe.model.config.max_position_embeddings,
            num_return_sequences=1,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        attentions = model_output.attentions
        model_output = pipe.tokenizer.decode(model_output.sequences[0])
        return model_output, attentions


if __name__ == "__main__":

    configs = MarkovConfigs.get_configs_from_cli_args()
    eval_engine = MarkovEvalEngine(configs)
    eval_engine.run()
