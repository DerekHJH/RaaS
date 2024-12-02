import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import torch

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

    # There are too many layers and heads, we construct the attention maps for
    # a limited number of layers and heads as configured in the `configs`.
    layer_ids = [0]
    head_ids = [0]


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

        self.construct_and_save_attention_maps(attentions)

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

    def construct_and_save_attention_maps(self, attentions: Tuple[Tuple[torch.Tensor]]) -> None:
        """
        Params:
            attentions: Tuple (of length `seq_len`) of Tuple (of length `num_layers`) of
            torch.Tensor --- `seq_len` * `num_layers` torch.Tensor in total,
            each of shape (`batch_size`, `num_heads`, `num_attend_tokens`, `num_attended_tokens`).

            assert len(attentions) == seq_len
            assert len(attentions[0]) == num_layers
            assert attentions[0][0].shape == (batch_size, num_heads, num_prefill_tokens, num_prefill_tokens)
            assert attentions[1][0].shape == (batch_size, num_heads, 1, num_prefill_tokens + 1)
            assert attentions[2][0].shape == (batch_size, num_heads, 1, num_prefill_tokens + 2)
            ...
        """
        import pdb

        pdb.set_trace()
        for layer_id in self.configs.layer_ids:
            for head_id in self.configs.head_ids:
                # A list (of length seq_len) torch.Tensor,
                # each with shape (num_attend_tokens, num_attended_tokens)
                attention = [
                    attentions[i][layer_id][0, head_id, :, :] for i in range(len(attentions))
                ]
                print(attention)

        attentions = []
        # In out test, batch_size is always 1
        attentions = attentions.squeeze(
            0
        )  # Shape (num_heads, num_attend_tokens, num_attended_tokens)


if __name__ == "__main__":

    configs = MarkovConfigs.get_configs_from_cli_args()
    eval_engine = MarkovEvalEngine(configs)
    eval_engine.run()
