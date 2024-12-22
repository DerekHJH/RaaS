import argparse
import logging
import os
import sys
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from tqdm.contrib import tenumerate
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    Pipeline,
    SinkCache,
    pipeline,
)

from benchmarks.data_sets.data_set import Data_set
from benchmarks.evals.utils import str2class

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfigs:
    dataset: str
    model: str
    approach: str
    tot_num_data: int = 200
    all_datasets: List[str] = field(
        default_factory=lambda: ["math500", "aime", "gsm8k"]
    )  # Fixed mutable default
    all_models: List[str] = field(
        default_factory=lambda: [
            "peiyi9979/mistral-7b-sft",
            "AIDC-AI/Marco-o1",
            "Qwen/Qwen2.5-Math-7B-Instruct",
        ]
    )
    all_approaches: List[str] = field(
        default_factory=lambda: [
            "full",
            "sink-64",
            "sink-128",
            "sink-256",
            "sink-512",
            "sink-1024",
            "quest-64",
            "quest-128",
            "quest-256",
            "quest-512",
            "quest-1024",
            "raas-64",
            "raas-128",
            "raas-256",
            "raas-512",
            "raas-1024",
        ]
    )

    seed: int = 42
    result_path: str = "results"

    @classmethod
    def get_configs_from_cli_args(cls) -> "EvalConfigs":
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

    def __init__(self, configs: EvalConfigs) -> None:
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
        self.model: AutoModelForCausalLM = self.load_model_for_approach(
            self.configs.model, self.configs.approach
        )
        self.pipe: Pipeline = self.load_pipeline(self.model, self.tokenizer)

        # Step 2: Run the inference and record results into the dataset
        self.dataset = self.run_inference(self.pipe, self.dataset)

        # Step 3: Generate the presentation
        self.generate_presentation()

    def load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """
        Load the tokenizer for the model.
        """

        logger.info(f"Loading the tokenizer \033[32m{model_name}\033[0m")

        # Avoid tokenization warnings (deadlock)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        return AutoTokenizer.from_pretrained(
            model_name,
            model_max_length=sys.maxsize,
            padding_side="right",
            trust_remote_code=True,
        )

    def load_dataset(self, dataset_name: str, tokenizer: AutoTokenizer) -> Data_set:
        """
        Load the dataset, finish preprocessing within the Data_set
        class and save the dataset.
        """
        logger.info(f"Loading the dataset \033[32m{dataset_name}\033[0m")
        dataset: Data_set = str2class[dataset_name](
            tokenizer=tokenizer,
            path=self.configs.result_path,
            tot_num_data=self.configs.tot_num_data,
        )
        dataset.save_dataset(self.configs.result_path)

        return dataset

    def load_model_for_approach(self, model_name: str, approach_name: str) -> AutoModelForCausalLM:
        """
        Load the model and decide on the type of KV cache.
        """

        logger.info(f"Loading the model \033[32m{model_name}\033[0m")

        model_config = AutoConfig.from_pretrained(model_name)
        if model_config.model_type == "llama":
            if approach_name == "full" or "sink" in approach_name:  # They differ only in cache type
                from quest.models.full_llama import LlamaForCausalLM

                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
            elif "quest" in approach_name:
                from quest.models.full_llama import LlamaForCausalLM
                from quest.models.quest_llama import enable_quest_attention_eval

                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
                enable_quest_attention_eval(
                    model,
                    {
                        "cache_budget": int(approach_name.split("-")[-1]),
                        "page_size": 16,  # Fixed as stated in the paper
                    },
                )
            elif "raas" in approach_name:
                from quest.models.raas_llama import (
                    LlamaForCausalLM,
                    enable_raas_attention_eval,
                )

                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
                enable_raas_attention_eval(
                    model,
                    {
                        "cache_budget": int(approach_name.split("-")[-1]),
                        "page_size": 16,  # Fixed as stated in the paper
                    },
                )
        elif model_config.model_type == "qwen2":
            if approach_name == "full" or "sink" in approach_name:  # They differ only in cache type

                from quest.models.full_qwen2 import Qwen2ForCausalLM

                model = Qwen2ForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
            elif "quest" in approach_name:
                from quest.models.full_qwen2 import Qwen2ForCausalLM
                from quest.models.quest_qwen2 import enable_quest_attention_eval

                model = Qwen2ForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
                enable_quest_attention_eval(
                    model,
                    {
                        "cache_budget": int(approach_name.split("-")[-1]),
                        "page_size": 16,  # Fixed as stated in the paper
                    },
                )
            elif "raas" in approach_name:
                from quest.models.raas_qwen2 import (
                    Qwen2ForCausalLM,
                    enable_raas_attention_eval,
                )

                model = Qwen2ForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
                enable_raas_attention_eval(
                    model,
                    {
                        "cache_budget": int(approach_name.split("-")[-1]),
                        "page_size": 16,  # Fixed as stated in the paper
                    },
                )

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
        for i, (prompt, answer) in tenumerate(dataset, desc="dataset", leave=True):
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
        elif "quest" in self.configs.approach:
            # Modifications happen on the model loading stage instead of here
            past_key_values = DynamicCache()  #  quest attention will not discard any cache
        elif "raas" in self.configs.approach:
            past_key_values = DynamicCache()

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
        """
        Present the results by invoking this function after executing run().
        Separating this function from run() improves efficiency by saving execution time.
        The results are saved in self.configs.result_path.
        """

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

    configs = EvalConfigs.get_configs_from_cli_args()
    eval_engine = EvalEngine(configs)
    eval_engine.run()
