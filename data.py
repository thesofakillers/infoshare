from datasets import load_dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from typing import Any, Callable, Dict, List, Tuple, Optional

import os


class UDDataModule(LightningDataModule):
    def __init__(
        self,
        treebank_name: str,
        tokenize_fn: Callable,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenize_fn"])

        self.tokenize_fn = tokenize_fn
        self.dataset_dir = os.path.join(data_dir, treebank_name)

    def prepare_data(self):
        if os.path.exists(self.dataset_dir):
            return

        print("Downloading UniversalDependencies data from HuggingFace")
        dataset = load_dataset("universal_dependencies", self.hparams.treebank_name)

        # remove all original columns except for tokens and upos
        columns = dataset["train"].column_names
        columns.remove("tokens")
        columns.remove("upos")

        print("Saving to disk")
        dataset.save_to_disk(self.dataset_dir)

    def setup(self, stage: Optional[str] = None):
        dataset = load_from_disk(self.dataset_dir)

        if stage == "fit" or stage is None:
            self.ud_train = dataset["train"]
            self.ud_val = dataset["validation"]

            self.num_classes = self.ud_train.info.features["upos"].feature.num_classes

        if stage == "test" or stage is None:
            self.ud_test = dataset["test"]

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[BatchEncoding, List[LongTensor]]:
        encodings = self.tokenize_fn([x["tokens"] for x in batch])
        target = [LongTensor(x["upos"]) for x in batch]
        return encodings, target

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ud_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ud_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ud_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn,
        )
