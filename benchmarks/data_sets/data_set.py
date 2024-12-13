import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class Data_set(ABC):
    """
    Use the underline to differentiate from huggingface
    datasets (dataset) and Datasets (Dataset).
    """

    def __init__(self, tokenizer, tot_num_data=int(1e6), path: str = None, **kwargs):
        """
        Load cached dataset. If failed, load raw data in the subclass.
        """
        self.tokenizer = tokenizer
        self.tot_num_data = tot_num_data
        self.path = path
        self.kwargs = kwargs
        self.data: pd.DataFrame = None

        if path is not None and os.path.exists(os.path.join(path, "data.json")):
            # Load processed data including partial evaluation results
            self.data = pd.read_json(os.path.join(path, "data.json"))
            logger.info(f"Loaded dataset from {path}")
        else:
            # Load raw data implemented by subclass
            self.data = self.load_raw_data()
            logger.info("Loaded dataset from raw data")

            # Common processing for all datasets
            self.data = self.data[:tot_num_data]
            self.data = self.data.apply(
                lambda row: self.create_groundtruth_field(row), axis=1
            ).apply(lambda row: self.create_prompt_field(row), axis=1)

    @abstractmethod
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw data from the subclass.

        Returns:
            The raw data in pandas DataFrame format.
        """
        raise NotImplementedError

    @abstractmethod
    def create_groundtruth_field(self, row: Dict) -> Dict:
        """
        Create the row[f'groundtruth'] column in the dataset. To be used with the apply function.

        Args:
            row: One row of the dataset.

        Returns:
            The row with the row[f'groundtruth'] updated.
        """
        raise NotImplementedError

    @abstractmethod
    def create_prompt_field(self, row: Dict) -> Dict:
        """
        Create the row[f'prompt'] column in the dataset. To be used with the apply function.

        Args:
            row: One row of the dataset.

        Returns:
            The row with the row[f'prompt'] updated.
        """
        raise NotImplementedError

    @abstractmethod
    def _calc_accuracy(self, row: Dict, approach: str) -> Dict:
        """
        Check the correctness of the model output and store the result in the f'accuracy_{approach}' column.
        And store the final output in the f'final_output_{approach}' column.

        Args:
            row: One row of the dataset.
            approach: The approach used to generate the output.

        Returns:
            The row with the row[f'accuracy_{approach}'], row[f'final_output_{approach}'] updated.
        """
        raise NotImplementedError

    # ALL following methods are common to all datasets.
    # SHOULD NOT BE OVERRIDDEN.
    # This is to ensure consistent interfaces.
    # If you have a strong need to override some of the following methods,
    # please refer to how calc_accruacy and _calc_accuracy methods are implemented.

    def calc_accuracy(self, approach: str) -> None:
        """
        Compare the model output with the groundtruth and calculate the accuracy.
        Store the accuracy in the f'accuracy_{approach}' column.

        Args:
            approach: The approach used to generate the output.

        """
        assert f"output_{approach}" in self.data.columns, f"output_{approach} not in the dataset"
        assert "groundtruth" in self.data.columns, "groundtruth not in the dataset"

        self.data = self.data.apply(lambda row: self._calc_accuracy(row, approach), axis=1)

    def update(self, new_data: Dict[str, List]) -> None:
        """
        Update the dataset with new dictionary data. If the length of the new data is less than the original data,
        fill the rest with None.

        Args:
            new_data: The new data to update the dataset.
        """
        for key, value in new_data.items():
            if len(value) < len(self.data):
                logger.warning(
                    (
                        "Length of new data is less than the original data: "
                        f"{len(value)} < {len(self.data)}"
                    )
                )
            self.data[key] = value + (len(self.data) - len(value)) * [None]

    def save_dataset(self, path: str) -> None:
        """
        Save the dataset to the given path.

        Args:
            path: The path to save the dataset.
        """
        self.data.to_json(os.path.join(path, "data.json"), orient="records", indent=4)

    def __iter__(self):
        """
        Iterate over the dataset, providing the prompt
        and groundtruths for the test driver.
        """
        for _, row in self.data.iterrows():
            yield row["prompt"], row["groundtruth"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]
