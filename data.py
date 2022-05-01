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
        pad_token: str,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        pbar: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenize_fn"])

        self.tokenize_fn = tokenize_fn
        self.dataset_dir = os.path.join(data_dir, treebank_name)

    def _map_wordpieces(self, tokens: List[List[str]]) -> List[List[int]]:
        encoding = self.tokenize_fn(tokens)

        def count_tokens(batch_idx: int) -> int:
            return (
                sum(x != self.hparams.pad_token for x in encoding.tokens(batch_idx)) - 2
            )  # -2 for [CLS] and [SEP]

        def count_pad(batch_idx: int) -> int:
            return sum(x == self.hparams.pad_token for x in encoding.tokens(batch_idx))

        return [
            [-1]  # [CLS]
            + [
                encoding.token_to_word(b_idx, token_idx)
                for token_idx in range(1, count_tokens(b_idx) + 1)
            ]  # tokens
            + [-1] * (count_pad(b_idx) + 1)  # [SEP] + [PAD]s
            for b_idx in range(len(tokens))
        ]

    def _add_pos_per_wordpiece(self, batch: Dict[str, list]) -> Dict[str, list]:
        batch["wmap"] = self._map_wordpieces(batch["tokens"])
        return batch

    def prepare_data(self):
        if os.path.exists(self.dataset_dir):
            return

        print("Downloading UniversalDependencies data from HuggingFace")
        dataset = load_dataset("universal_dependencies", self.hparams.treebank_name)

        # remove all original columns except for tokens and upos
        columns = dataset["train"].column_names
        columns.remove("tokens")
        columns.remove("upos")

        print("Preprocessing data (adding POS per wordpiece)")
        dataset = dataset.map(
            self._add_pos_per_wordpiece, batched=True, remove_columns=columns
        )

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

    def collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[List[List[BatchEncoding]], List[LongTensor], List[LongTensor]]:
        encodings = self.tokenize_fn([x["tokens"] for x in batch])
        token_pos = [LongTensor(x["upos"]) for x in batch]
        wordpiece_map = [LongTensor(x["wmap"]) for x in batch]
        return encodings, token_pos, wordpiece_map

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
