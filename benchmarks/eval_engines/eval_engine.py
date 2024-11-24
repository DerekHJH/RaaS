import os
import argparse
from dataclasses import dataclass
import dataclasses
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

@dataclass
class Configs:

    dataset: str
    model: str
    approach: str
    all_datasets: List[str] = ['needle', 'math500'] # 'math', 
    all_models: List[str] = ['peiyi9979/mistral-7b-sft']
    all_approaches: List[str] = ['full', 'quest'] # 'RaaS'
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
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        configs = cls(**{attr: getattr(args, attr) for attr in attrs})
        return configs
        

    def __post_init__(self):
        self._verify_init_args()
        self.result_path = os.path.join(self.result_path, self.dataset, self.model.split('/')[-1])

    def _verify_init_args(self):
        assert self.model in self.all_models, f'{self.model} not in {self.all_models}'
        assert self.dataset in self.all_datasets, f'{self.dataset} not in {self.all_datasets}'
        assert self.approach in self.all_approaches, f'{self.approach} not in {self.all_approaches}'
    

class EvalEngine:
    
    def __init__(self, configs: Configs) -> None:
        self.configs = configs

    def run(self):
        logging.info(f'Running {self.configs.approach} on {self.configs.dataset} using {self.configs.model}')
        logging.info(f'Saving the results to {self.configs.result_path}')
        
        

