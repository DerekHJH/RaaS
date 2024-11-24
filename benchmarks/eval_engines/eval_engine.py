import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
logging.basicConfig(level=logging.INFO)

# The following imports are subject to change
from evaluation.llama import enable_tuple_kv_cache_for_llama
from evaluation.mistral import enable_tuple_kv_cache_for_mistral

@dataclass
class Configs:

    dataset: str
    model: str
    approach: str
    all_datasets: List[str] = field(default_factory=lambda: ['needle', 'math500'])  # Fixed mutable default
    all_models: List[str] = field(default_factory=lambda: ['peiyi9979/mistral-7b-sft'])  # Fixed mutable default
    all_approaches: List[str] = field(default_factory=lambda: ['full', 'quest'])  # Fixed mutable default
    seed: int = 42
    result_path: str = 'results'

    @classmethod
    def get_configs_from_cli_args(cls) -> 'Configs':
        """
        Parse the command line arguments and return the Configs object.
        """
        # Add the arguments to the parser.
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument('--model', type=str, required=True)
        parser.add_argument('--approach', type=str, required=True)
        parser.add_argument('--seed', type=int, default=42)
        
        # Parse the arguments.
        args = parser.parse_args()
        configs = cls(**vars(args))
        return configs
        

    def __post_init__(self):
        self._verify_init_args()
        self.result_path = os.path.join(self.result_path, self.dataset, self.model.split('/')[-1])
        os.makedirs(self.result_path, exist_ok=True)

    def _verify_init_args(self):
        assert self.model in self.all_models, f'{self.model} not in {self.all_models}'
        assert self.dataset in self.all_datasets, f'{self.dataset} not in {self.all_datasets}'
        assert self.approach in self.all_approaches, f'{self.approach} not in {self.all_approaches}'
    

class EvalEngine:
    
    def __init__(self, configs: Configs) -> None:
        self.configs = configs

    def run(self):
        logging.info(f'Running \033[32m{self.configs.approach}\033[0m on \033[32m{self.configs.dataset}\033[0m using \033[32m{self.configs.model}\033[0m')
        logging.info(f'Saving the results to \033[32m{self.configs.result_path}\033[0m')

        self._run_inference()

        self._calc_metrics()

        self._plot_figures()


    def _run_inference(self):
        # Avoid tokenization warnings (deadlock)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.configs.model,
            model_max_length=sys.maxsize,
            padding_side="right",
            trust_remote_code=True,
        )

        torch.cuda.empty_cache()

        if 'llama' in self.configs.model.lower() or 'longchat' in self.configs.model.lower():
            enable_tuple_kv_cache_for_llama()
        if 'mistral' in self.configs.model.lower():
            enable_tuple_kv_cache_for_mistral()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.configs.model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if self.configs.approach == 'quest':
            from evaluation.quest_attention import enable_quest_attention_eval
            # enable_quest_attention_eval(self.model, args)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
            

    def _calc_metrics(self):
        pass

    def _plot_figures(self):
        pass
        

