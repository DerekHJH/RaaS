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
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

from benchmarks.data_sets.data_set import Data_set
from benchmarks.eval_engines.utils import str2class
from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Configs:

    dataset: str
    model: str
    approach: str
    tot_num_data: int = int(1e6)
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

        # Step 1: Preprocessing, load and modify neccessary components such as
        # tokenizer, dataset, model and pipeline.
        self.tokenizer: AutoTokenizer = self.load_tokenizer(self.configs.model)
        self.dataset: Data_set = self.load_dataset(self.configs.dataset, self.tokenizer)
        self.model: AutoModelForCausalLM = self.load_model(self.configs.model)
        self.model: AutoModelForCausalLM = self.modify_model_according_to_approach(
            self.model, self.configs.approach
        )
        self.pipe: Pipeline = self.load_pipeline(self.model, self.tokenizer)

        # Step 2: Run the inference and record results into the dataset
        self.dataset = self.run_inference(self.pipe, self.dataset)

    def load_tokenizer(self, tokenizer: str) -> AutoTokenizer:
        """
        Load the tokenizer for the model.
        """

        logger.info(f"Loading the tokenizer \033[32m{tokenizer}\033[0m")

        # Avoid tokenization warnings (deadlock)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        return AutoTokenizer.from_pretrained(
            tokenizer,
            model_max_length=sys.maxsize,
            padding_side="right",
            trust_remote_code=True,
        )

    def load_dataset(self, dataset: str, tokenizer: AutoTokenizer) -> Data_set:
        """
        Load the dataset, finish preprocessing within the Data_set
        class and save the dataset.
        """
        logger.info(f"Loading the dataset \033[32m{dataset}\033[0m")
        dataset = str2class[dataset](
            tokenizer=tokenizer,
            path=self.configs.result_path,
            tot_num_data=self.configs.tot_num_data,
        )
        dataset.save_dataset(self.configs.result_path)

        return dataset

    def load_model(self, model: str) -> AutoModelForCausalLM:
        """
        Load the model.

        Before loading the model, we need to enable the tuple_kv_cache
        for quest BC. The current huggingface kv cache is implemented
        as Cache class https://huggingface.co/docs/transformers/main/en/kv_cache
        """

        logger.info(f"Loading the model \033[32m{model}\033[0m")

        if "llama" in model.lower() or "longchat" in model.lower():
            enable_tuple_kv_cache_for_llama()
        if "mistral" in model.lower():
            enable_tuple_kv_cache_for_mistral()

        return AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    def modify_model_according_to_approach(
        self, model: AutoModelForCausalLM, approach: str
    ) -> AutoModelForCausalLM:
        """
        Reload the model according to the approach.
        """
        logger.info(f"Reload the model according to the approach \033[32m{approach}\033[0m")
        if approach == "quest":
            from evaluation.quest_attention import enable_quest_attention_eval

            enable_quest_attention_eval(model, self.configs)
        elif approach == "raas":
            from evaluation.raas_attention import enable_raas_attention_eval

            enable_raas_attention_eval(model, self.configs)
        else:  # The "full" approach
            pass
        return model

    def load_pipeline(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> Pipeline:
        """
        Assemble the pipeline with the model and the tokenizer.
        """
        logger.info("Use a pipeline to aggregate the model and the tokenizer")
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
        )

    def run_inference(self, pipe: Pipeline, dataset: Data_set) -> Data_set:
        """
        Run the inference and record the results into the dataset.
        """
        logger.info("Run the inference. This might take a long time... Good luck")
        results = defaultdict(list)
        for i, (prompt, answer) in tenumerate(dataset, desc="dataset", leave=False):
            model_output, TTFT, JCT, TPOT, num_decode = self.test_model(pipe, prompt, answer)
            results[f"output_{self.configs.approach}"].append(model_output)
            # TODO: Also record the time-related metrics
            results[f"TTFT_{self.configs.approach}"].append(TTFT)
            results[f"JCT_{self.configs.approach}"].append(JCT)
            results[f"TPOT_{self.configs.approach}"].append(TPOT)
            results[f"num_decode_{self.configs.approach}"].append(num_decode)
        dataset.update(results)
        dataset.calc_accuracy(self.configs.approach)
        dataset.save_dataset(self.configs.result_path)

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

        return dataset

    def test_model(self, pipe, prompt, answer) -> Tuple[str, float, float, float, int]:
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
