import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from tqdm.contrib import tenumerate
from transformers import DynamicCache, Pipeline, SinkCache

from benchmarks.data_sets.data_set import Data_set
from benchmarks.eval_engines.eval_engine import Configs, EvalEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class E2EConfigs(Configs):
    # Overriding the default values of the parent class.
    tot_num_data: int = int(1e6)
    all_datasets: List[str] = field(
        default_factory=lambda: ["math500", "AIME"]
    )  # Fixed mutable default
    all_models: List[str] = field(
        default_factory=lambda: [
            "peiyi9979/mistral-7b-sft",
            "AIDC-AI/Marco-o1",
            "Qwen/Qwen2.5-Math-7B-Instruct",
        ]
    )
    all_approaches: List[str] = field(default_factory=lambda: ["full", "quest", "streamingllm"])


class E2EEvalEngine(EvalEngine):

    def run_inference(self, pipe: Pipeline, dataset: Data_set) -> Data_set:

        logger.info("Run the inference. This might take a long time... Good luck")
        results = defaultdict(list)
        for i, (prompt, answer) in tenumerate(dataset, desc="dataset", leave=False):
            model_output, TTFT, JCT, TPOT, num_decode = self.test_model(pipe, prompt, answer)
            results[f"output_{self.configs.approach}"].append(model_output)
            results[f"TTFT_{self.configs.approach}"].append(TTFT)
            results[f"JCT_{self.configs.approach}"].append(JCT)
            results[f"TPOT_{self.configs.approach}"].append(TPOT)
            results[f"num_decode_{self.configs.approach}"].append(num_decode)
        dataset.update(results)
        dataset.save_dataset(self.configs.result_path)

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
            logger.warning(f"No chat template found. Using the prompt as is.")
            extended_prompt = prompt
        inputs = pipe.tokenizer(extended_prompt, return_tensors="pt").to("cuda:0")
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        cache_position = torch.arange(input_ids.shape[1], dtype=torch.int64, device="cuda:0")
        if self.configs.approach == "full":
            past_key_values = DynamicCache()
        elif self.configs.approach == "streamingllm":
            past_key_values = SinkCache(window_length=128, num_sink_tokens=4)

        with torch.no_grad():

            # Prefill
            start_time = time.perf_counter()
            output = pipe.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                use_cache=True,
            )
            prefill_time = time.perf_counter() - start_time

            next_token_id = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [next_token_id.item()]

            # Decode autoregressively
            decode_time = []
            for num_decode in range(
                pipe.model.config.max_position_embeddings - 512
            ):  # Reserve 1024 tokens for the prompt

                input_ids = next_token_id
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
                cache_position = cache_position[-1:] + 1

                start_time = time.perf_counter()
                outputs = pipe.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                decode_time.append(time.perf_counter() - start_time)

                # Produece the next token
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [next_token_id.item()]

                if next_token_id.item() == pipe.tokenizer.eos_token_id:
                    break

            TTFT = prefill_time
            JCT = prefill_time + np.sum(decode_time)
            TPOT = np.sum(decode_time) / num_decode

        model_output = pipe.tokenizer.decode(generated_content, skip_special_tokens=True)
        return model_output, TTFT, JCT, TPOT, num_decode

    def generate_presentation(self):

        self.dataset.calc_accuracy(self.configs.approach)
        self.dataset.save_dataset(self.configs.result_path)

        accuracy_avg = np.mean(self.dataset.data[f"accuracy_{self.configs.approach}"])
        TTFT_avg = np.mean(self.dataset.data[f"TTFT_{self.configs.approach}"])
        JCT_avg = np.mean(self.dataset.data[f"JCT_{self.configs.approach}"])
        TPOT_avg = np.mean(self.dataset.data[f"TPOT_{self.configs.approach}"])
        num_decode_avg = np.mean(self.dataset.data[f"num_decode_{self.configs.approach}"])
        logger.info(f"Average accuracy of {self.configs.approach}: {accuracy_avg:.3f}")
        logger.info(f"Average TTFT of {self.configs.approach}: {TTFT_avg:.2f} s")
        logger.info(f"Average JCT of {self.configs.approach}: {JCT_avg:.2f} s")
        logger.info(f"Average TPOT of {self.configs.approach}: {TPOT_avg:.2f} s")
        logger.info(f"Average num_decode of {self.configs.approach}: {num_decode_avg:.2f}")


if __name__ == "__main__":

    configs = E2EConfigs.get_configs_from_cli_args()
    eval_engine = E2EEvalEngine(configs)
    eval_engine.run()
