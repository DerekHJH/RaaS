from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List
import os
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class Data_set(ABC):
    """
    Use the underline to differentiate from huggingface 
    datasets (dataset) and Datasets (Dataset).
    """
    def __init__(
            self, 
            tokenizer,
            tot_num_data=int(1e6), 
            path: str=None,
            **kwargs
        ):
        """
        Load cached dataset. If failed, load raw data in the subclass.
        """
        self.tokenizer = tokenizer
        self.tot_num_data = tot_num_data
        self.path = path
        self.kwargs = kwargs
        self.data: pd.DataFrame = None

        if path is not None and os.path.exists(os.path.join(path, 'data.json')):
            # Load processed data including partial evaluation results
            self.data = pd.read_json(os.path.join(path, 'data.json'))
            logger.info(f'Loaded dataset from {path}')
        else:
            # Load raw data implemented by subclass
            self.data = self.load_raw_data()
            logger.info('Loaded dataset from raw data')

            # Common processing for all datasets
            self.data = self.data[:tot_num_data]
            self.data = self.data.apply(lambda row: self._create_answer_field(row), axis=1) \
                .apply(lambda row: self._create_prompt_field(row), axis=1)

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _create_answer_field(self, row: Dict) -> Dict:
        raise NotImplementedError
    
    @abstractmethod
    def _create_prompt_field(self, row: Dict) -> Dict:
        raise NotImplementedError

    def update(self, new_data: Dict[str, List]) -> None:
        for key, value in new_data.items():
            if len(value) < len(self.data):
                logger.warning(f'Length of new data is less than the original data: {len(value)} < {len(self.data)}')
            self.data[key] = value + (len(self.data) - len(value)) * [None]

    def save_dataset(self, path: str) -> None:
        self.data.to_json(
            os.path.join(path, 'data.json'),
            orient="records", 
            indent=4
        )

    def calc_accuracy(self, approach: str) -> None:
        """
        Compare the model output with the answer and calculate the accuracy.
        Store the accuracy in the f'accuracy_{approach}' column.
        """
        assert f'output_{approach}' in self.data.columns, f'output_{approach} not in the dataset'
        assert 'answer' in self.data.columns, 'answer not in the dataset'

        self.data = self.data.apply(lambda row: self._calc_accuracy(row, approach), axis=1)
    
    @abstractmethod
    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:
        raise NotImplementedError

    def __iter__(self):
        """
        Iterate over the dataset, providing the prompt and answers for the test driver.
        """
        for _, row in self.data.iterrows():
            yield row['prompt'], row['answer']
            
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data.iloc[idx]
