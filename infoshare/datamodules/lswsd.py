"""DataModule for Lexical Sample task"""
from collections import defaultdict
import os
from typing import Callable, Dict, List, Optional, Tuple, Any

import datasets
from datasets import Features, Value, Sequence, ClassLabel
from datasets.arrow_dataset import Dataset
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from torch import LongTensor
import numpy as np
import pandas as pd

from infoshare.datamodules.base import BaseDataModule
from infoshare.utils import just_underscore, just_zero, list_of_zero


class LSWSDDataModule(BaseDataModule):
    """
    Lexical sampling of word senses in SemCor
    """

    train: Dataset
    val: Dataset
    test: Dataset

    pos_id2cname: List[str]
    pos_cname2id: Dict[str, int]
    id_to_cname: List[str]
    cname_to_id: Dict[str, int]
    num_classes: int

    def __init__(
        self,
        task: str,
        tokenize_fn: Callable,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        """Data module for SemCor.

        Args:
            task (str): the task to train the probing classifier on (LSWSD or POS).
            tokenize_fn (Callable): a func takes sentence and returns list of tokens
            data_dir (str): the data directory to load/store the datasets
            batch_size (int): the batch size used by the dataloaders
            num_workers (int): the number of subprocesses used by the dataloaders
        """
        super().__init__()
        self.save_hyperparameters(ignore=["tokenize_fn"])

        valid_tasks = ["LSWSD", "POS"]
        assert task in valid_tasks, f"Task must be one of {valid_tasks}"

        self.tokenize_fn = tokenize_fn
        self.data_dir = os.path.join(data_dir, "semcor")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, task, "processed")

        # fmt: off
        self.UD_POS_TAGS = [
            "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET",
            "CCONJ", "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX",
        ]
        # https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
        self.penn_to_ud = defaultdict(
            just_underscore,
            { # conversion to JSON format courtesy of ChatGPT
                "#": "SYM", "$": "SYM", "''": "PUNCT", ",": "PUNCT", "-LRB-": "PUNCT",
                "-RRB-": "PUNCT", ".": "PUNCT", ":": "PUNCT", "AFX": "ADJ", "CC":
                "CCONJ", "CD": "NUM", "DT": "DET", "EX": "PRON", "FW": "X", "HYPH":
                "PUNCT", "IN": "ADP", "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ", "LS":
                "X", "MD": "VERB", "NIL": "X", "NN": "NOUN", "NNP": "PROPN", "NNPS":
                "PROPN", "NNS": "NOUN", "PDT": "DET", "POS": "PART", "PRP": "PRON",
                "PRP$": "DET", "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "RP": "ADP",
                "SYM": "SYM", "TO": "PART", "UH": "INTJ", "VB": "VERB", "VBD": "VERB",
                "VBG": "VERB", "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB", "WDT":
                "DET", "WP": "PRON", "WP$": "DET", "WRB": "ADV", "``": "PUNCT",
            },
        )
        # fmt: on
        self.pos_id2cname = self.UD_POS_TAGS
        self.pos_cname2id = {tag: i for i, tag in enumerate(self.pos_id2cname)}
        self.has_setup = False
        # these are the lemmas we are interested in
        with open("lswsd_lemmas.txt", "r") as f:
            self.lemmas = set(f.read().splitlines())

    def prepare_data(self) -> None:
        """Takes care of downloading and preparing data"""
        if self.has_setup:
            return
        if os.path.exists(self.data_dir):
            print("Dataset already downloaded.")
            self.raw_dset_dict = datasets.load_from_disk(self.raw_dir)
            return

        print("Downloading SemCor from HuggingFace")
        semcor = datasets.load_dataset("thesofakillers/SemCor")

        print("Performing minor preprocessing")
        # put them all together
        dataset = datasets.concatenate_datasets([semcor[i] for i in semcor])
        # pandas preprocessing
        df = dataset.to_pandas()
        # where our actual labels will come from
        df["sense"] = df["lemma"] + "%" + df["lexsn"]
        # grouping by sentence: each row is a sentence rather than a word now
        sentence_df = df.groupby(["tagfile", "pnum", "snum"]).agg(
            tokens=pd.NamedAgg(column="value", aggfunc=list),
            idxs=pd.NamedAgg(column="lemma", aggfunc=self._where_salient),
            lemmas=pd.NamedAgg(column="lemma", aggfunc=list),
            senses=pd.NamedAgg(column="sense", aggfunc=list),
            pos=pd.NamedAgg(column="pos", aggfunc=list),
        )
        # remove sentences with no salient words
        sentence_df = sentence_df[sentence_df["idxs"].apply(len) > 0]
        # back to HF dataset format for shuffling, splitting and serializing
        sentence_df = sentence_df.reset_index().drop(
            columns=["tagfile", "pnum", "snum"]
        )
        sentence_df["id"] = sentence_df.index
        dataset = datasets.Dataset.from_pandas(sentence_df)
        # shuffle
        dataset = dataset.shuffle(seed=42)
        # train, val, test split (80%, 10%, 10%)
        trainval_test_dict = dataset.train_test_split(test_size=0.1, seed=42)
        train_val_dict = trainval_test_dict["train"].train_test_split(
            test_size=1 / 9, seed=42
        )
        self.raw_dset_dict = datasets.DatasetDict(
            {
                "train": train_val_dict["train"],
                "val": train_val_dict["test"],
                "test": trainval_test_dict["test"],
            }
        )
        print("Saving to disk")
        os.makedirs(self.raw_dir, exist_ok=True)
        self.raw_dset_dict.save_to_disk(self.raw_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.has_setup:
            return
        # processing so that the correct columns are available for our classifier
        if self.hparams.task == "LSWSD":
            self.id_to_cname = set()
            self.train = self.process_for_lswsd(
                self.raw_dset_dict["train"], split="train"
            )
            self.val = self.process_for_lswsd(self.raw_dset_dict["val"], split="val")
            self.test = self.process_for_lswsd(self.raw_dset_dict["test"], split="test")
        elif self.hparams.task == "POS":
            self.train = self.process_for_pos(
                self.raw_dset_dict["train"], split="train"
            )
            self.val = self.process_for_pos(self.raw_dset_dict["val"], split="val")
            self.test = self.process_for_pos(self.raw_dset_dict["test"], split="test")
        else:
            raise ValueError(f"Unknown task {self.hparams.task}")

    def process_for_pos(self, dset: Dataset, split: str) -> Dataset:
        processed_path = os.path.join(self.processed_dir, split)

        if os.path.exists(processed_path):
            proc_dataset = datasets.load_from_disk(processed_path)
            if split == "train":
                self._handle_cname_maps_pos()
            return proc_dataset

        print(f"Processing {split} split...")

        # get rid of unnecessary columns
        sentences = dset.remove_columns(["lemmas", "senses", "idxs"])

        if split == "train":
            self._handle_cname_maps_pos()

        sentences = sentences.map(self._convert_to_ids_pos)

        # need to create a new dataset to set our own features (lame)
        proc_dataset = Dataset.from_dict(
            sentences.to_dict(),
            features=Features(
                {
                    "id": Value("int64"),
                    "tokens": Sequence(Value("string")),
                    "pos": Sequence(
                        ClassLabel(num_classes=self.num_classes, names=self.id_to_cname)
                    ),
                }
            ),
        )

        print("Saving to disk")
        os.makedirs(processed_path, exist_ok=True)
        proc_dataset.save_to_disk(processed_path)

        print("Done.")
        return proc_dataset

    def process_for_lswsd(self, dset: Dataset, split: str) -> Dataset:
        """
        Processes the raw dataset so to only consider the lexical senses that
        have been selected. The resulting dataset contains the following columns.
        One row per sentence
        - tokens: the words in the sentence
        - senses: the ids of the senses of the words in the sentence
        - idxs: the indexes of the sense-annotated words in the sentence
        - pos: the ids of the POS tags of the sense-annotated words in the sentence
        - lemmas: the lemmas of the sense-annotated words in the sentence
        """
        processed_path = os.path.join(self.processed_dir, split)

        # don't need to do the remaining processing if we've already done it
        if os.path.exists(processed_path):
            proc_dataset = datasets.load_from_disk(processed_path)
            if split == "train":
                self.id_to_cname = proc_dataset.features["senses"].feature.names
                self._handle_cname_maps_lswsd()
            return proc_dataset

        print(f"Processing {split} split...")

        # get rid of features not directly linked to the salient idxs (tokens untouched)
        sentences = dset.map(self._reduce_to_salients)

        if split == "train":
            for sentence in sentences:
                for sense in sentence["senses"]:
                    self.id_to_cname.add(sense)
            self.id_to_cname = list(self.id_to_cname)
            self.id_to_cname.insert(0, "unk")
            self._handle_cname_maps_lswsd()

        sentences = sentences.map(self._convert_to_ids_lswsd)

        # need to create a new dataset to set our own features (lame)
        proc_dataset = Dataset.from_dict(
            sentences.to_dict(),
            features=Features(
                {
                    "id": Value("int64"),
                    "tokens": Sequence(Value("string")),
                    "lemmas": Sequence(Value("string")),
                    "idxs": Sequence(Value("int64")),
                    "senses": Sequence(ClassLabel(names=self.id_to_cname)),
                    "pos": Sequence(
                        ClassLabel(
                            num_classes=len(self.UD_POS_TAGS), names=self.pos_id2cname
                        )
                    ),
                }
            ),
        )

        print("Saving to disk")
        os.makedirs(processed_path, exist_ok=True)
        proc_dataset.save_to_disk(processed_path)

        print("Done.")
        return proc_dataset

    def _convert_to_ids_pos(self, input_row):
        pos_ids = [self.cname_to_id[self.penn_to_ud[pos]] for pos in input_row["pos"]]
        input_row["pos"] = pos_ids
        return input_row

    def _convert_to_ids_lswsd(self, input_row):
        pos_ids = [self.pos_cname2id[self.penn_to_ud[pos]] for pos in input_row["pos"]]
        sense_ids = [self.cname_to_id[sense] for sense in input_row["senses"]]
        input_row["pos"] = pos_ids
        input_row["senses"] = sense_ids
        return input_row

    def _handle_cname_maps_pos(self):
        self.id_to_cname = self.pos_id2cname
        self.cname_to_id = {cname: idx for idx, cname in enumerate(self.id_to_cname)}
        self.num_classes = len(self.id_to_cname)

    def _handle_cname_maps_lswsd(self):
        """Add 'unk' to the beginning of the list of sense names and computes inverse"""
        self.cname_to_id = defaultdict(
            just_zero, {sense: i for i, sense in enumerate(self.id_to_cname)}
        )
        lemma_to_sense_ids = {}
        self.sense_id_to_lemma = ["" for _ in range(len(self.id_to_cname))]
        for sense in self.id_to_cname[1:]:  # skipping 'unk'
            lemma = sense.split("%")[0]
            if lemma not in lemma_to_sense_ids:
                lemma_to_sense_ids[lemma] = []
            sense_id = self.cname_to_id[sense]
            lemma_to_sense_ids[lemma].append(sense_id)
            self.sense_id_to_lemma[sense_id] = lemma
        self.lemma_to_sense_ids = defaultdict(list_of_zero, lemma_to_sense_ids)
        self.sense_id_to_lemma[0] = "unk"
        self.num_classes = len(self.id_to_cname)

    def _reduce_to_salients(self, input_row):
        """Keeps only the senses and pos relevant to our lexical samples"""
        new_lemmas = []
        new_senses = []
        new_pos = []
        for i in input_row["idxs"]:
            new_lemmas.append(input_row["lemmas"][i])
            new_senses.append(input_row["senses"][i])
            new_pos.append(input_row["pos"][i])
        input_row["lemmas"] = new_lemmas
        input_row["senses"] = new_senses
        input_row["pos"] = new_pos
        return input_row

    def _where_salient(self, agg_input):
        return np.where(agg_input.isin(self.lemmas))[0]

    def get_collate_fn(self):
        """
        Parses an incoming batch of data, tokenizing the sentences in the process
        """
        if self.hparams.task == "LSWSD":
            return self.wsd_collate_fn
        elif self.hparams.task == "POS":
            return self.pos_collate_fn
        else:
            raise ValueError(f"Unknown task {self.hparams.task}")

    def pos_collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[BatchEncoding, List[LongTensor]]:
        """Custom collate function for the POS task."""
        encodings = self.tokenize_fn([x["tokens"] for x in batch])
        targets = [LongTensor(x["pos"]) for x in batch]
        return encodings, targets

    def wsd_collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[
        BatchEncoding, List[LongTensor], List[LongTensor], List[LongTensor], List[str]
    ]:
        """Custom collate function for the LSWSD task."""
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
            self.train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.get_collate_fn(),
        )
