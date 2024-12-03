import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap

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
    layer_ids = list(range(32))
    head_ids = list(range(32))


class MarkovEvalEngine(EvalEngine):

    def run_inference(self, pipe: Pipeline, dataset: Data_set) -> Data_set:
        """
        Run the inference and record the results into the dataset.
        """
        results = defaultdict(list)
        for i, (prompt, answer) in tenumerate(dataset, desc="dataset", leave=False):
            model_output, attentions = self.test_model(pipe, prompt, answer)
            results[f"output_{self.configs.approach}"].append(model_output)

        dataset.update(results)
        dataset.save_dataset(self.configs.result_path)
        torch.save(attentions, os.path.join(self.configs.result_path, "attentions.pt"))

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

    def present_results(self) -> None:
        """
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
        attentions: Tuple[Tuple[torch.Tensor]] = torch.load(
            self.configs.result_path + "attentions.pt"
        )

        for layer_id in self.configs.layer_ids:
            for head_id in self.configs.head_ids:
                # A list (of length seq_len) torch.Tensor,
                # each with shape (num_attend_tokens, num_attended_tokens)
                attention = [
                    attentions[i][layer_id][0, head_id, :, :] for i in range(len(attentions))
                ]
                """
                assert attention[0].shape == (num_prefill_tokens, num_prefill_tokens)
                assert attention[1].shape == (1, num_prefill_tokens + 1)
                assert attention[2].shape == (1, num_prefill_tokens + 2)
                """
                assert attention[0].shape[0] == attention[0].shape[1]
                assert attention[1].shape == (1, attention[0].shape[1] + 1)
                assert attention[2].shape == (1, attention[0].shape[1] + 2)

                for i, tensor in enumerate(attention):
                    padding = (0, attention[-1].shape[1] - tensor.shape[1])  # (left, right)
                    attention[i] = torch.nn.functional.pad(
                        tensor, padding, mode="constant", value=0
                    )

                attention = torch.cat(attention, dim=0).cpu().float()  # shape (seq_len, seq_len)

                # Min-max normalization for better visibility
                attention = (attention - attention.min(dim=-1).values) / (
                    attention.max(dim=-1).values - attention.min(dim=-1).values
                )

                plt.figure(figsize=(12, 10))
                red_black_cmap = LinearSegmentedColormap.from_list("RedBlack", ["black", "red"])
                sns.heatmap(attention, cmap=red_black_cmap)
                plt.savefig(self.configs.result_path + f"layer_{layer_id}_head_{head_id}.png")
                plt.close()


if __name__ == "__main__":

    configs = MarkovConfigs.get_configs_from_cli_args()
    eval_engine = MarkovEvalEngine(configs)
    # eval_engine.run()
    eval_engine.present_results()
