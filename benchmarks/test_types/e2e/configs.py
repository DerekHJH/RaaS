import os
import logging
logging.basicConfig(level=logging.INFO)

class Configs:

    all_datasets = ['needle'] # 'math', 
    all_models = ['peiyi9979/mistral-7b-sft']
    all_approaches = ['quest'] # 'RaaS'

    def __init__(self, 
        dataset=all_datasets[0], 
        model=all_models[0], 
        approach=all_approaches[0]
    ):

        self.dataset = dataset
        self.model = model
        self.approach = approach  
        self._verify_init_args()
        

        # Create parent folders for results
        self.result_folder = os.path.join('results', dataset, model.split('/')[-1])
        os.makedirs(self.result_folder, exist_ok=True)

        # Other configs
        self.seed = 42

    def _verify_init_args(self):
        assert self.model in self.all_models, f'{self.model} not in {self.all_models}'
        assert self.dataset in self.all_datasets, f'{self.dataset} not in {self.all_datasets}'
        assert self.approach in self.all_approaches, f'{self.approach} not in {self.all_approaches}'