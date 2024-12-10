import argparse
import logging
import os
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    pipeline,
)

from benchmarks.data_sets.data_set import Data_set
from benchmarks.eval_engines.utils import str2class

# from quest.utils.cache_utils import Cache, SinkCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Configs:

    dataset: str
    model: str
    approach: str
    tot_num_data: int = int(1e6)
    all_datasets: List[str] = field(default_factory=lambda: ["math500"])
    all_models: List[str] = field(default_factory=lambda: ["peiyi9979/mistral-7b-sft"])
    all_approaches: List[str] = field(default_factory=lambda: ["full", "quest", "raas"])
    seed: int = 42
    result_path: str = "results"

    # Quest configs
    page_size: int = 16  # Also RaaS config
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
        self.model: AutoModelForCausalLM = self.load_model_for_approach(
            self.configs.model, self.configs.approach
        )
        self.pipe: Pipeline = self.load_pipeline(self.model, self.tokenizer)

        # Step 2: Run the inference and record results into the dataset
        self.dataset = self.run_inference(self.pipe, self.dataset)

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

        Before loading the model, we need to enable the tuple_kv_cache
        for quest BC. The current huggingface kv cache is implemented
        as Cache class https://huggingface.co/docs/transformers/main/en/kv_cache
        """

        logger.info(f"Loading the model \033[32m{model_name}\033[0m")

        model_config = AutoConfig.from_pretrained(model_name)
        if model_config.model_type == "llama":
            if approach_name == "full":
                from transformers import LlamaForCausalLM

                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
            elif approach_name == "streamingllm":
                # Use the same llama code as the full model
                # TODO:
                pass
                # from quest.models.full_llama import LlamaForCausalLM

                # model = LlamaForCausalLM.from_pretrained(
                #     model_name,
                #     device_map="cuda:0",
                #     trust_remote_code=True,
                # )
            elif approach_name == "quest":
                from quest.models.quest_llama_new import LlamaForCausalLM

                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
                model.quest_init(
                    page_size=self.configs.page_size,
                    max_seq_len=model.config.max_position_embeddings,
                    token_budget=self.configs.token_budget,
                    dtype=torch.float16,
                    device=torch.device("cuda:0"),
                )
            elif approach_name == "raas":
                from quest.models.raas_llama import LlamaForCausalLM

                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    device_map="cuda:0",
                    trust_remote_code=True,
                )
                # TODO: Finish initialization

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

    @abstractmethod
    def run_inference(self, pipe: Pipeline, dataset: Data_set) -> Data_set:
        """
        Run the inference and record the results into the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_presentation(self):
        """
        Present the results by invoking this function after executing run().
        Separating this function from run() improves efficiency by saving execution time.
        The results are saved in self.configs.result_path.
        """
        raise NotImplementedError
