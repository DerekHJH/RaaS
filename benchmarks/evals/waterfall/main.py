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
from transformers import DynamicCache, Pipeline, SinkCache

from benchmarks.data_sets.data_set import Data_set
from benchmarks.evals.e2e.main import EvalConfigs, EvalEngine

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class MarkovConfigs(EvalConfigs):
    # Overriding the default values of the parent class.
    tot_num_data: int = 1


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
        torch.save(
            attentions,
            os.path.join(self.configs.result_path, f"{self.configs.approach}_attentions.pt"),
        )

        return dataset

    def test_model(
        self, pipe: Pipeline, prompt: str, answer: str
    ) -> Tuple[str, float, float, float, int]:

        torch.cuda.empty_cache()
        # Prepare the input
        try:
            extended_prompt = pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            logger.debug(f"No chat template found. Using the prompt as is.")
            extended_prompt = prompt
        inputs = pipe.tokenizer(extended_prompt, return_tensors="pt").to("cuda:0")
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64, device="cuda:0")

        # Initialize the cache
        if self.configs.approach == "full":
            past_key_values = DynamicCache()
        elif "sink" in self.configs.approach:
            cache_budget = int(self.configs.approach.split("-")[-1])
            past_key_values = SinkCache(window_length=cache_budget, num_sink_tokens=4)
        elif "h2o" in self.configs.approach:
            from quest.utils.cache_utils import H2OCache

            cache_budget = int(self.configs.approach.split("-")[-1])
            past_key_values = H2OCache(cache_budget=cache_budget)
        elif "quest" in self.configs.approach:
            # Modifications happen on the model loading stage instead of here
            past_key_values = DynamicCache()  #  quest attention will not discard any cache
        elif "raas" in self.configs.approach:
            from quest.utils.cache_utils import RaaSCache

            cache_budget = int(self.configs.approach.split("-")[-1])
            past_key_values = RaaSCache(page_size=16, cache_budget=cache_budget)
        attentions = None
        with torch.no_grad():

            # Prefill
            outputs = pipe.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )

            attentions = list(outputs.attentions)

            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [next_token_id.item()]

            # Decode autoregressively
            for num_decode in range(
                pipe.model.config.max_position_embeddings - 512
            ):  # Reserve 1024 tokens for the prompt

                input_ids = next_token_id
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
                cache_position = cache_position[-1:] + 1

                output = pipe.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )

                attns = list(output.attentions)
                for layer_id, (tot_attn, row_attn) in enumerate(zip(attentions, attns)):
                    """
                    tot_attn: shape (bsz, num_heads, seq_len, seq_len) with bsz == 1
                    row_attn: shape (bsz, num_heads, 1, seq_len + 1) with bsz == 1
                    """
                    assert (
                        tot_attn.shape[0] == 1 and row_attn.shape[0] == 1
                    ), "Support batch size 1 only"
                    assert tot_attn.shape[1] == row_attn.shape[1], "num_heads mismatch"
                    assert (
                        tot_attn.shape[2] == tot_attn.shape[3]
                        and tot_attn.shape[2] + 1 == row_attn.shape[3]
                    ), "seq_len mismatch"

                    padding = (0, 1)  # (left, right)
                    tot_attn = torch.nn.functional.pad(tot_attn, padding, mode="constant", value=0)
                    tot_attn = torch.cat([tot_attn, row_attn], dim=2)
                    attentions[layer_id] = tot_attn

                # Produece the next token
                next_token_id = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [next_token_id.item()]

                if next_token_id.item() == pipe.tokenizer.eos_token_id:
                    break

        model_output = pipe.tokenizer.decode(generated_content, skip_special_tokens=True)
        return model_output, attentions

    def generate_presentation(self) -> None:
        pass


if __name__ == "__main__":

    configs = MarkovConfigs.get_configs_from_cli_args()
    eval_engine = MarkovEvalEngine(configs)
    eval_engine.run()
