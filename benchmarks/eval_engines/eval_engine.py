import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from tqdm.contrib import tenumerate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from benchmarks.eval_engines.utils import str2class
from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class Configs:

    dataset: str
    model: str
    approach: str
    tot_num_data: int = 3
    all_datasets: List[str] = field(default_factory=lambda: ["needle", "math500"])
    all_models: List[str] = field(default_factory=lambda: ["peiyi9979/mistral-7b-sft"])
    all_approaches: List[str] = field(default_factory=lambda: ["full", "quest", "raas"])
    seed: int = 42
    result_path: str = "results"

    # Quest configs
    chunk_size: int = 16  # Also RaaS config
    token_budget: int = 1024

    @classmethod
    def get_configs_from_cli_args(cls) -> "Configs":
        """
        Parse the command line arguments and return the Configs object.
        """
        # Add the arguments to the parser.
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--model", type=str, required=True)
        parser.add_argument("--approach", type=str, required=True)
        parser.add_argument("--seed", type=int, default=42)

        # Parse the arguments.
        args = parser.parse_args()
        configs = cls(**vars(args))
        return configs

    def __post_init__(self):
        """
        Verify the init arguments and create the result path.
        """
        self._verify_init_args()
        self.result_path = os.path.join(self.result_path, self.dataset, self.model.split("/")[-1])
        os.makedirs(self.result_path, exist_ok=True)

    def _verify_init_args(self):
        assert self.model in self.all_models, f"{self.model} not in {self.all_models}"
        assert self.dataset in self.all_datasets, f"{self.dataset} not in {self.all_datasets}"
        assert self.approach in self.all_approaches, f"{self.approach} not in {self.all_approaches}"


class EvalEngine:
    """
    Evaluate a specific approach on a specific model and a specific dataset.
    """

    def __init__(self, configs: Configs) -> None:
        self.configs = configs

    def run(self):
        logging.info(
            (
                f"Evaluate \033[32m{self.configs.approach}\033[0m on"
                f" \033[32m{self.configs.model}\033[0m and"
                f" \033[32m{self.configs.dataset}\033[0m"
            )
        )
        logging.info(f"Save the results to \033[32m{self.configs.result_path}\033[0m")

        logger.debug("Step 1: Load the tokenizer")
        # Avoid tokenization warnings (deadlock)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model,
            model_max_length=sys.maxsize,
            padding_side="right",
            trust_remote_code=True,
        )

        logger.debug("Step 2: Load the dataset, preprocess the data and save the preprocced data")
        self.dataset = str2class[self.configs.dataset](
            tokenizer=self.tokenizer,
            path=self.configs.result_path,
            tot_num_data=self.configs.tot_num_data,
        )
        # ckpt 1: dataset preprocessed
        self.dataset.save_dataset(self.configs.result_path)

        logger.debug("Step 3: Load the model")
        if "llama" in self.configs.model.lower() or "longchat" in self.configs.model.lower():
            enable_tuple_kv_cache_for_llama()
        if "mistral" in self.configs.model.lower():
            enable_tuple_kv_cache_for_mistral()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.configs.model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        logger.debug("Step 4: Reload the model according to the approach")
        if self.configs.approach == "quest":
            from evaluation.quest_attention import enable_quest_attention_eval

            enable_quest_attention_eval(self.model, self.configs)
        elif self.configs.approach == "raas":
            from evaluation.raas_attention import enable_raas_attention_eval

            enable_raas_attention_eval(self.model, self.configs)
        else:  # The "full" approach
            pass

        logger.debug("Step 5: Assemble the pipeline with the model and the tokenizer")
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        logger.debug("Step 6: Run the inference and record results")
        results = defaultdict(list)
        for i, (prompt, answer) in tenumerate(self.dataset, desc="dataset", leave=False):
            model_output, TTFT, JCT, TPOT, num_decode = self._test_model(self.pipe, prompt, answer)
            results[f"output_{self.configs.approach}"].append(model_output)
            # TODO: Also record the time-related metrics
            results[f"TTFT_{self.configs.approach}"].append(TTFT)
            results[f"JCT_{self.configs.approach}"].append(JCT)
            results[f"TPOT_{self.configs.approach}"].append(TPOT)
            results[f"num_decode_{self.configs.approach}"].append(num_decode)

        logger.debug("Step 7: Save the results")
        self.dataset.update(results)

        # ckpt 2: dataset augmented with inference results
        self.dataset.save_dataset(self.configs.result_path)

        logger.debug("Step 8: Calculate the accuracy for the model outputs.")
        self.dataset.calc_accuracy(self.configs.approach)

        # ckpt 3: dataset augmented with accuracy
        self.dataset.save_dataset(self.configs.result_path)

        # Print some aggregate information
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

    def _test_model(self, pipe, prompt, answer) -> Tuple[str, float, float, float, int]:
        # model_output = pipe(prompt,
        # num_return_sequences=1)[0]["generated_text"][len(prompt_text):]

        input = pipe.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():

            start_time = time.perf_counter()

            # Prefill
            output = pipe.model(
                input_ids=input.input_ids,
                past_key_values=None,
                use_cache=True,
            )

            torch.cuda.synchronize()
            prefill_time = time.perf_counter() - start_time

            # Store KV cache
            past_key_values = output.past_key_values

            # Produce the first token
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]

            decode_time = 0
            # Decode autoregressively
            for num_decode in range(pipe.model.config.max_position_embeddings - 1):

                start_time = time.perf_counter()

                outputs = pipe.model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                torch.cuda.synchronize()
                decode_time += time.perf_counter() - start_time

                # Store KV cache
                past_key_values = outputs.past_key_values

                # Produece the next token
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]

                if pred_token_idx.item() == pipe.tokenizer.eos_token_id:
                    break

            TTFT = prefill_time
            JCT = prefill_time + decode_time
            TPOT = decode_time / num_decode

        model_output = pipe.tokenizer.decode(generated_content, skip_special_tokens=True)
        return model_output, TTFT, JCT, TPOT, num_decode
