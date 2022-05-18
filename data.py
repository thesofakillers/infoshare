from argparse import ArgumentParser
from datasets import load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from typing import Any, Callable, Dict, List, Tuple, Optional

import os


class UDDataModule(LightningDataModule):
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
            default="POS",
            choices=["DEP", "POS"],
            help="The task to train the probing classifier on.",
        )

        parser.add_argument(
            "--treebank_name",
            type=str,
            default="en_gum",
            help="The name of the treebank to use as the dataset.",
        )

        return parent_parser

    def __init__(
        self,
        task: str,
        treebank_name: str,
        tokenize_fn: Callable,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenize_fn"])

        valid_tasks = ["POS", "DEP"]
        assert task in valid_tasks, f"Task must be one of {valid_tasks}."

        self.tokenize_fn = tokenize_fn
        self.dataset_dir = os.path.join(data_dir, treebank_name)

    def prepare_data(self):
        if os.path.exists(self.dataset_dir):
            return

        print("Downloading UniversalDependencies data from HuggingFace")
        dataset = load_dataset("universal_dependencies", self.hparams.treebank_name)

        # Only keep columns that are relevant to our task
        keep_columns = ["tokens"]
        if self.hparams.task == "POS":
            keep_columns += ["upos"]
        elif self.hparams.task == "DEP":
            keep_columns += ["head", "deprel"]

        # Remove tokens that correspond to the underscore class.
        # The UniversalDependencies dataset essentially splits compound words into their parts
        # but keeps the original token in the list of tokens as well. This will cause problems
        # when we try to parse the sentences, especially for dependency relations which is
        # affected by the word order as the "head" value is a token index.
        if self.hparams.task == "POS":
            target_class = "upos"
            target_value = dataset.info.features["upos"].names.index("_")
        else:
            target_class = "deprel"
            target_value = "_"

        def remove_underscores(x: Dict[str, list]) -> Dict[str, list]:
            keep_indices = [
                idx for idx, value in enumerate(x[target_class]) if value != target_value
            ]
            for column in keep_columns:
                x[column] = [x[column][idx] for idx in keep_indices]

            return x

        drop_columns = set(dataset["train"].column_names).difference(keep_columns)
        dataset = dataset.map(remove_underscores, remove_columns=list(drop_columns))

        print("Saving to disk")
        dataset.save_to_disk(self.dataset_dir)

    def setup(self, stage: Optional[str] = None):
        dataset = load_from_disk(self.dataset_dir)

        if stage == "fit" or stage is None:
            self.ud_train = dataset["train"]
            self.ud_val = dataset["validation"]

            if self.hparams.task == "POS":
                self.id_to_cname = self.ud_train.info.features["upos"].feature.names
                self.num_classes = self.ud_train.info.features["upos"].feature.num_classes
            elif self.hparams.task == "DEP":
                # Aggregate all classes from the train dataset
                # We include "_" to comply with the number of classes in the dataset spec
                all_classes = set("_").union(*self.ud_train["deprel"])
                # Remove classes that have a language-specific modifier
                main_classes = sorted([drel for drel in all_classes if ":" not in drel])
                # Create a mapping from (main) class names to unique ids
                self.cname_to_id = {cname: i for i, cname in enumerate(main_classes)}
                # Add a mapping from the language-specific classes to the broader ones
                for spec_class in all_classes.difference(main_classes):
                    col_idx = spec_class.index(":")
                    main_class = spec_class[:col_idx]
                    self.cname_to_id[spec_class] = self.cname_to_id[main_class]

                self.id_to_cname = main_classes
                self.num_classes = len(main_classes)

        if stage == "test" or stage is None:
            self.ud_test = dataset["test"]

        if stage == "debug" or stage is None:
            self.ud_debug = dataset["validation"].select(list(range(50)))

    def get_collate_fn(self) -> Callable:
        if self.hparams.task == "POS":
            return self.pos_collate_fn
        elif self.hparams.task == "DEP":
            return self.dep_collate_fn
        # Add more cases here if needed

    def map_drels_to_ids(self, drels: List[str]) -> LongTensor:
        return LongTensor([self.cname_to_id[drel] for drel in drels])

    def pos_collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[BatchEncoding, List[LongTensor]]:
        encodings = self.tokenize_fn([x["tokens"] for x in batch])
        targets = [LongTensor(x["upos"]) for x in batch]
        return encodings, targets

    def dep_collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[BatchEncoding, List[LongTensor], List[LongTensor]]:
        encodings = self.tokenize_fn([x["tokens"] for x in batch])
        heads = [LongTensor([int(h) for h in x["head"]]) for x in batch]
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
