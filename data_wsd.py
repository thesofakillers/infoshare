"Work in progress"

from functools import partial
import os
import zipfile
from argparse import ArgumentParser
from typing import Callable, Optional, Dict, List

from datasets.arrow_dataset import Dataset
from pytorch_lightning import LightningDataModule
from transformers.models.auto.tokenization_auto import AutoTokenizer

from utils import download_and_unzip


class WSDDataModule(LightningDataModule):
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
        """Data module for the Raganato Benchmark.

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
        self.dataset_dir = os.path.join(data_dir, "wsd")

        self.train_dir_name = "WSD_Training_Corpora"
        self.eval_dir_name = "WSD_Unified_Evaluation_Datasets"

    def prepare_data(self) -> None:
        """Takes care of downloading data"""
        if os.path.exists(self.dataset_dir):
            print("Dataset already downloaded.")
            return

        os.makedirs(self.dataset_dir, exist_ok=True)
        print("train data")
        download_and_unzip(
            f"http://lcl.uniroma1.it/wsdeval/data/{self.train_dir_name}.zip",
            self.dataset_dir,
            f"{self.train_dir_name}.zip",
        )
        print("eval data")
        download_and_unzip(
            f"http://lcl.uniroma1.it/wsdeval/data/{self.eval_dir_name}.zip",
            self.dataset_dir,
            f"{self.eval_dir_name}.zip",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    WSDDataModule.add_model_specific_args(parser)
    parser.add_argument("--encoder_name", default="roberta-base", type=str)
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, add_prefix_space=True)
    tokenize_fn = partial(
        tokenizer,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
    )
    wsd = WSDDataModule(
        args.task,
        tokenize_fn,
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )
    wsd.prepare_data()
