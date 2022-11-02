"Raganati et al. (2017) data for WSD task"
from argparse import ArgumentParser
from datasets.arrow_dataset import Dataset
from functools import partial
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from typing import Callable, Optional, Dict, List, Tuple, Any
from utils import download_and_unzip

import datasets
import numpy as np
import os
import pandas as pd
import xml.etree.ElementTree as ET


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

    # Declare dataset fields
    semcor: Dataset
    semeval2007: Dataset
    semeval2013: Dataset
    semeval2015: Dataset
    senseval2: Dataset
    senseval3: Dataset
    wsd_debug: Dataset
    # Declare label maps
    pos_id2cname: List[str]
    pos_cname2id: Dict[str, int]
    sense_id2cname: List[str]
    sense_cname2id: Dict[str, int]

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
        self.data_dir = os.path.join(data_dir, "wsd")

        self.train_dir = "WSD_Training_Corpora"
        self.eval_dir = "WSD_Unified_Evaluation_Datasets"

        self.POS_TAGS = [
            "NOUN",
            "PUNCT",
            "ADP",
            "NUM",
            "SYM",
            "SCONJ",
            "ADJ",
            "PART",
            "DET",
            "CCONJ",
            "PROPN",
            "PRON",
            "X",
            "_",
            "ADV",
            "INTJ",
            "VERB",
            "AUX",
        ]

    def prepare_data(self) -> None:
        """Takes care of downloading data"""
        if os.path.exists(self.data_dir):
            print("Dataset already downloaded.")
            return

        os.makedirs(self.data_dir, exist_ok=True)
        print("Downloading train data")
        download_and_unzip(
            f"http://lcl.uniroma1.it/wsdeval/data/{self.train_dir}.zip",
            self.data_dir,
            f"{self.train_dir}.zip",
        )
        print("Downloading eval data")
        download_and_unzip(
            f"http://lcl.uniroma1.it/wsdeval/data/{self.eval_dir}.zip",
            self.data_dir,
            f"{self.eval_dir}.zip",
        )

    def setup(self, stage: Optional[str] = None):
        """Sets up data for our model"""
        # ignore stage and setup all datasets, since we need id2cname from train for all
        self.semcor = self.wsd_dset(os.path.join(self.data_dir, self.train_dir, "SemCor"), True)
        self.semeval2007 = self.wsd_dset(os.path.join(self.data_dir, self.eval_dir, "semeval2007"))
        self.semeval2013 = self.wsd_dset(os.path.join(self.data_dir, self.eval_dir, "semeval2013"))
        self.semeval2015 = self.wsd_dset(os.path.join(self.data_dir, self.eval_dir, "semeval2015"))
        self.senseval2 = self.wsd_dset(os.path.join(self.data_dir, self.eval_dir, "senseval2"))
        self.senseval3 = self.wsd_dset(os.path.join(self.data_dir, self.eval_dir, "senseval3"))
        self.wsd_debug = self.wsd_dset(
            os.path.join(self.data_dir, self.train_dir, "SemCor")
        ).select(list(range(50)))

    def init_label_maps(self, senses: List[str]):
        """Initialize the label maps used by the dataloaders in the collate_fn"""
        self.pos_id2cname = self.POS_TAGS
        self.pos_cname2id = {cname: i for i, cname in enumerate(self.pos_id2cname)}
        self.sense_id2cname = senses
        self.sense_cname2id = {cname: i for i, cname in enumerate(self.sense_id2cname)}

    def wsd_dset(self, dataset_path: str, is_train: bool = False) -> Dataset:
        """
        Given path to Raganato benchmark XML and gold labels
        Returns a torch dataset or hf dataset
        """
        processed_path = os.path.join(dataset_path, "processed")
        dataset_name = dataset_path.split("/")[-1]
        print(f"Creating dataset for {dataset_name}")
        if os.path.exists(processed_path):
            print("Dataset already processed. Loading from disk")
            dataset = datasets.load_from_disk(processed_path)

            # initialize label maps during train set
            if is_train:
                senses = dataset.features["senses"].feature.names
                self.init_label_maps(senses)

            return dataset

        gold_path = os.path.join(dataset_path, f"{dataset_name.lower()}.gold.key.txt")
        with open(gold_path, "r") as f:
            gold_labels = f.readlines()
            gold_labels = [tuple(line.strip().split(" ")[:2]) for line in gold_labels]
            gold_labels = pd.DataFrame(gold_labels, columns=["token_id", "label"])
            gold_labels = gold_labels.set_index("token_id")

        # label maps populated during training
        if is_train:
            senses = ["unk"] + np.random.RandomState(seed=42).permutation(
                gold_labels.label.unique()
            ).tolist()
            self.init_label_maps(senses)

        # parse XML
        data_tree = ET.parse(f"{dataset_path}/{dataset_name.lower()}.data.xml")
        data_root = data_tree.getroot()
        data = []
        for text in data_root:
            for sentence in text:
                sent_words = []
                sent_lemmas = []
                idxs = []
                senses = []
                pos = []
                for idx, word in enumerate(sentence):
                    sent_words.append(word.text)
                    sent_lemmas.append(word.get("lemma"))
                    if word.tag == "instance":
                        # keep track of where instances occur
                        idxs.append(idx)
                        pos.append(self.pos_cname2id[word.get("pos")])
                        senses.append(
                            self.sense_cname2id.get(
                                gold_labels.loc[word.get("id")].item(), 0
                            )  # unk's position in mapping is 0
                        )
                data.append((sentence.get("id"), sent_words, sent_lemmas, idxs, senses, pos))
        # convert to dataframe
        data_df = pd.DataFrame(data, columns=["id", "tokens", "lemmas", "idxs", "senses", "pos"])
        data_df = data_df.set_index("id")
        # and finally to hf dataset
        dataset = Dataset.from_pandas(
            data_df,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "lemmas": datasets.Sequence(datasets.Value("string")),
                    "idxs": datasets.Sequence(datasets.Value("int64")),
                    "senses": datasets.Sequence(datasets.ClassLabel(names=self.sense_id2cname)),
                    "pos": datasets.Sequence(
                        datasets.ClassLabel(num_classes=len(self.POS_TAGS), names=self.pos_id2cname)
                    ),
                }
            ),
        )
        os.makedirs(processed_path, exist_ok=True)
        dataset.save_to_disk(processed_path)
        return dataset

    def get_collate_fn(self):
        if self.hparams.task == "WSD":
            return self.wsd_collate_fn

    def wsd_collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[BatchEncoding, List[LongTensor], List[LongTensor], List[LongTensor], List[str]]:
        encodings = self.tokenize_fn([x["tokens"] for x in batch])
        target_senses = [LongTensor(x["senses"]) for x in batch]
        target_idxs = [LongTensor(x["idxs"]) for x in batch]
        target_pos = [LongTensor(x["pos"]) for x in batch]
        lemmas = [x["lemmas"] for x in batch]
        # for the following lists:
        # - len(encodings) > len(lemmas)  -- due to tokenization
        # - len(target_senses) == len(target_idxs) == len(target_pos)
        return encodings, target_senses, target_idxs, target_pos, lemmas

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.semcor,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.semeval2007,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
        )

    def test_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                collate_fn=self.get_collate_fn(),
            )
            for dataset in [
                self.semeval2013,
                self.semeval2015,
                self.senseval2,
                self.senseval3,
            ]
        ]

    def debug_dataloader(self) -> DataLoader:
        return DataLoader(
            self.wsd_debug,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
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
    wsd.setup()
