import os
from typing import Any, Callable, Dict, List, Tuple, Optional
from collections import defaultdict
import xml.etree.ElementTree as ET

from datasets import load_from_disk, Features, Sequence, Value, ClassLabel
from torch import LongTensor
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from transformers import BatchEncoding
import pandas as pd
import numpy as np

from infoshare.datamodules.base import BaseDataModule
from infoshare.utils import download_and_unzip, list_of_zero, just_zero


class WSDDataModule(BaseDataModule):
    "Raganato et al. (2017) benchmark data for WSD task"

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
    id_to_cname: List[str]
    cname_to_id: Dict[str, int]
    lemma_to_sense_ids: Dict[str, List[int]]
    num_classes: int
    idx_to_dataset: List[str]

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
        self.has_setup = False

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
        if self.has_setup:
            return
        self.semcor = self.wsd_dset(
            os.path.join(self.data_dir, self.train_dir, "SemCor"), True
        )
        self.semeval2007 = self.wsd_dset(
            os.path.join(self.data_dir, self.eval_dir, "semeval2007")
        )
        self.semeval2013 = self.wsd_dset(
            os.path.join(self.data_dir, self.eval_dir, "semeval2013")
        )
        self.semeval2015 = self.wsd_dset(
            os.path.join(self.data_dir, self.eval_dir, "semeval2015")
        )
        self.senseval2 = self.wsd_dset(
            os.path.join(self.data_dir, self.eval_dir, "senseval2")
        )
        self.senseval3 = self.wsd_dset(
            os.path.join(self.data_dir, self.eval_dir, "senseval3")
        )
        self.wsd_debug = self.wsd_dset(
            os.path.join(self.data_dir, self.train_dir, "SemCor")
        ).select(list(range(50)))
        self.has_setup = True

    def init_label_maps(self, gold_labels: pd.core.frame.DataFrame):
        """Initialize the label maps used by the dataloaders in the collate_fn"""
        self.pos_id2cname = self.POS_TAGS
        self.pos_cname2id = {cname: i for i, cname in enumerate(self.pos_id2cname)}
        senses: List[str] = ["unk"] + np.random.RandomState(seed=42).permutation(
            gold_labels.label.unique()
        ).tolist()
        self.id_to_cname = senses
        self.num_classes = len(senses)
        self.cname_to_id = {cname: i for i, cname in enumerate(self.id_to_cname)}
        self.cname_to_id = defaultdict(just_zero, self.cname_to_id)
        gold_labels["sense_id"] = gold_labels.label.map(self.cname_to_id)
        gold_labels["lemma"] = gold_labels["label"].str.split("%").str[0]
        self.lemma_to_sense_ids = (
            gold_labels.groupby("lemma")["sense_id"].apply(list).to_dict()
        )
        self.lemma_to_sense_ids = defaultdict(list_of_zero, self.lemma_to_sense_ids)
        # mapping test dataset idxs to dataset names
        self.idx_to_dataset = ["semeval2013", "semeval2015", "senseval2", "senseval3"]

    def wsd_dset(self, dataset_path: str, is_train: bool = False) -> Dataset:
        """
        Given path to Raganato benchmark XML and gold labels
        Returns a torch dataset or hf dataset
        """
        processed_path = os.path.join(dataset_path, "processed")
        dataset_name = dataset_path.split("/")[-1]

        gold_path = os.path.join(dataset_path, f"{dataset_name.lower()}.gold.key.txt")
        with open(gold_path, "r") as f:
            gold_labels = f.readlines()
        gold_labels = [tuple(line.strip().split(" ")[:2]) for line in gold_labels]
        gold_labels = pd.DataFrame(gold_labels, columns=["token_id", "label"])
        gold_labels = gold_labels.set_index("token_id")

        # label maps populated during training
        if is_train:
            self.init_label_maps(gold_labels)

        # don't need to do the remaining processing if we've already done it
        if os.path.exists(processed_path):
            dataset = load_from_disk(processed_path)
            return dataset
        print(f"Processing dataset: {dataset_name}")

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
                    if word.tag == "instance":
                        # keep track of where instances occur
                        sent_lemmas.append(word.get("lemma"))
                        idxs.append(idx)
                        pos.append(self.pos_cname2id[word.get("pos")])
                        senses.append(
                            self.cname_to_id[gold_labels["label"].loc[word.get("id")]]
                        )
                data.append(
                    (sentence.get("id"), sent_words, sent_lemmas, idxs, senses, pos)
                )
        # convert to dataframe
        data_df = pd.DataFrame(
            data, columns=["id", "tokens", "lemmas", "idxs", "senses", "pos"]
        )
        data_df = data_df.set_index("id")
        # and finally to hf dataset
        dataset = Dataset.from_pandas(
            data_df,
            features=Features(
                {
                    "id": Value("string"),
                    "tokens": Sequence(Value("string")),
                    "lemmas": Sequence(Value("string")),
                    "idxs": Sequence(Value("int64")),
                    "senses": Sequence(ClassLabel(names=self.id_to_cname)),
                    "pos": Sequence(
                        ClassLabel(
                            num_classes=len(self.POS_TAGS), names=self.pos_id2cname
                        )
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
    ) -> Tuple[
        BatchEncoding, List[LongTensor], List[LongTensor], List[LongTensor], List[str]
    ]:
        encodings = self.tokenize_fn([x["tokens"] for x in batch])
        target_senses = [LongTensor(x["senses"]) for x in batch]
        target_idxs = [LongTensor(x["idxs"]) for x in batch]
        target_pos = [LongTensor(x["pos"]) for x in batch]
        target_lemmas = [x["lemmas"] for x in batch]
        # for the following lists:
        # - len(encodings) > everything else, since we need every word for context emb
        # - len(target_senses) == len(target_idxs) == len(target_pos) == len(target_lemmas)
        return encodings, target_senses, target_idxs, target_pos, target_lemmas

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
