"Work in progress"

import os
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Callable, Tuple

from datasets.arrow_dataset import Dataset
from datasets.load import load_dataset
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from transformers.tokenization_utils_base import BatchEncoding


class SemCorDataModule(LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Dataset")
        parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="The batch size used by the dataloaders.",
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="./data",
            help="The data directory to load/store the datasets.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="The number of subprocesses used by the dataloaders.",
        )
        parser.add_argument(
            "--task",
            type=str,
            default="WSD",
            choices=["WSD"],
            help="The task to train the probing classifier on.",
        )
        return parent_parser

        # Declare variables that will be initialized later
        train: Dataset
        val: Dataset
        test: Dataset
        debug: Dataset
        cname_to_id: Optional[Dict[str, int]]
        id_to_cname: List[str]
        num_classes: int

        def __init__(
            self,
            task: str,
            tokenize_fn: Callable,
            data_dir: str = "./data",
            batch_size: int = 64,
            num_workers: int = 4,
        ):
            """Data module for the Universal Dependencies framework.

            Args:
                task (str): the task to train the probing classifier on (WSD).
                tokenize_fn (Callable): a func takes sentence and returns list of tokens
                data_dir (str): the data directory to load/store the datasets
                batch_size (int): the batch size used by the dataloaders
                num_workers (int): the number of subprocesses used by the dataloaders
            """
            super().__init__()
            self.save_hyperparameters(ignore=["tokenize_fn"])

            valid_tasks = ["WSD"]
            assert task in valid_tasks, f"Task must be one of {valid_tasks}."

            self.tokenize_fn = tokenize_fn
            self.dataset_dir = os.path.join(data_dir, "semcor")

    def prepare_data(self) -> None:
        if os.path.exists(self.dataset_dir):
            return

        print("Downloading SemCor data from HuggingFace")
        dataset = load_dataset("thesofakillers/SemCor")

        # Only keep columns that are relevant to our tasks
        keep_columns = ["lemma", "value", "lexsn", "wnsn"]

        drop_columns = set(dataset["brown1"].column_names).difference(keep_columns)
        dataset = dataset.remove_columns(drop_columns)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[BatchEncoding, List[LongTensor]]:
        """Custom collate function for the per-word WSD task"""
        encodings = self.tokenize_fn([x["lemma"] for x in batch])
        targets = [LongTensor(x["lexsn"]) for x in batch]
        return encodings, targets
