import os
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Tuple, Optional

from datasets import load_dataset, load_from_disk
from torch import LongTensor
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import BatchEncoding

from infoshare.datamodules.base import BaseDataModule


class UDDataModule(BaseDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Universal Dependencies")
        parser.add_argument(
            "--treebank_name",
            type=str,
            default="en_gum",
            help="The name of the treebank to use as the dataset.",
        )
        return parent_parser

    # Declare variables that will be initialized later
    ud_train: Dataset
    ud_val: Dataset
    ud_test: Dataset
    ud_debug: Dataset
    cname_to_id = Optional[Dict[str, int]]
    id_to_cname: List[str]
    num_classes: int

    def __init__(
        self,
        task: str,
        treebank_name: str,
        tokenize_fn: Callable,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        """Data module for the Universal Dependencies framework.

        Args:
            task (str): the task to train the probing classifier on (either POS or DEP).
            treebank_name (str): the name of the treebank to use as the dataset
            tokenize_fn (Callable): a function that takes a sentence and returns a list of tokens
            data_dir: (str): the data directory to load/store the datasets
            batch_size (int): the batch size used by the dataloaders
            num_workers (int): the number of subprocesses used by the dataloaders
        """
        super().__init__()
        self.save_hyperparameters(ignore=["tokenize_fn"])

        valid_tasks = ["POS", "DEP"]
        assert task in valid_tasks, f"Task must be one of {valid_tasks}."

        self.tokenize_fn = tokenize_fn
        self.dataset_dir = os.path.join(data_dir, treebank_name)

    def prepare_data(self):
        if os.path.exists(self.dataset_dir):
            print("Dataset already downloaded.")
            return

        print("Downloading UniversalDependencies data from HuggingFace")
        dataset = load_dataset("universal_dependencies", self.hparams.treebank_name)

        # Only keep columns that are relevant to our tasks
        keep_columns = ["tokens", "upos", "head", "deprel"]

        # Remove tokens that correspond to the underscore class.
        # The UniversalDependencies dataset essentially splits compound words into their parts
        # but keeps the original token in the list of tokens as well. This will cause problems
        # when we try to parse the sentences, especially for dependency relations which is
        # affected by the word order as the "head" value is a token index.
        def remove_underscores(x: Dict[str, list]) -> Dict[str, list]:
            keep_indices = [
                idx for idx, value in enumerate(x["head"]) if value != "None"
            ]
            for column in keep_columns:
                x[column] = [x[column][idx] for idx in keep_indices]

            return x

        drop_columns = set(dataset["train"].column_names).difference(keep_columns)
        dataset = dataset.map(remove_underscores, remove_columns=list(drop_columns))

        # Handle the "head" features being a string instead of an int and some "deprel" features
        # having a language-specific relation modifier.
        def handle_dep_features(x: Dict[str, list]) -> Dict[str, list]:
            x["head"] = [int(value) for value in x["head"]]
            x["deprel"] = [value.split(":")[0] for value in x["deprel"]]
            return x

        dataset = dataset.map(handle_dep_features)

        print("Saving to disk")
        dataset.save_to_disk(self.dataset_dir)

    def setup(self, stage: Optional[str] = None):
        dataset = load_from_disk(self.dataset_dir)

        if stage == "fit" or stage is None:
            self.ud_train = dataset["train"]
            self.ud_val = dataset["validation"]

            if self.hparams.task == "POS":
                self.id_to_cname = self.ud_train.info.features["upos"].feature.names
                self.num_classes = self.ud_train.info.features[
                    "upos"
                ].feature.num_classes
            elif self.hparams.task == "DEP":
                # Aggregate all classes from the train dataset
                # We include "_" to comply with the number of classes in the dataset spec
                main_classes = sorted(set("_").union(*self.ud_train["deprel"]))
                # Create a mapping from class names to unique ids
                self.cname_to_id = {cname: i for i, cname in enumerate(main_classes)}

                self.id_to_cname = main_classes
                self.num_classes = len(main_classes)

        if stage == "test" or stage is None:
            self.ud_test = dataset["test"]

        if stage == "debug" or stage is None:
            self.ud_debug = dataset["validation"].select(list(range(50)))

    def get_collate_fn(self) -> Callable:
        """Returns a collate function for the dataloader based on the task."""
        if self.hparams.task == "POS":
            return self.pos_collate_fn
        elif self.hparams.task == "DEP":
            return self.dep_collate_fn
        # Add more cases here if needed

    def map_drels_to_ids(self, drels: List[str]) -> LongTensor:
        """Maps a list of dependency relations to unique ids."""
        return LongTensor([self.cname_to_id[drel] for drel in drels])

    def pos_collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[BatchEncoding, List[LongTensor]]:
        """Custom collate function for the POS task."""
        encodings = self.tokenize_fn([x["tokens"] for x in batch])
        targets = [LongTensor(x["upos"]) for x in batch]
        return encodings, targets

    def dep_collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[BatchEncoding, List[LongTensor], List[LongTensor]]:
        """Custom collate function for the DEP task."""
        encodings = self.tokenize_fn([x["tokens"] for x in batch])
        heads = [LongTensor(x["head"]) for x in batch]
        targets = [self.map_drels_to_ids(x["deprel"]) for x in batch]
        return encodings, heads, targets

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ud_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ud_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ud_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
        )

    def debug_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ud_debug,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
        )
