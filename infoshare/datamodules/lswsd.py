"""DataModule for Lexical Sample task"""
from collections import defaultdict
import os
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

import datasets
from datasets import Features, Value, Sequence, ClassLabel
from datasets.arrow_dataset import Dataset
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from torch import LongTensor
from tqdm import tqdm
import numpy as np
import pandas as pd

from infoshare.datamodules.base import BaseDataModule


class LSWSDDataModule(BaseDataModule):
    """
    Lexical sampling of SemCor
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
        """Data module for the Lexical Sampling of SemCor.

        Args:
            task (str): the task to train the probing classifier on (WSD).
            tokenize_fn (Callable): a func takes sentence and returns list of tokens
            data_dir (str): the data directory to load/store the datasets
            batch_size (int): the batch size used by the dataloaders
            num_workers (int): the number of subprocesses used by the dataloaders
        """
        super().__init__()
        self.save_hyperparameters(ignore=["tokenize_fn"])

        valid_tasks = ["LSWSD"]
        assert task in valid_tasks, f"Task must be one of {valid_tasks}"

        self.tokenize_fn = tokenize_fn
        self.data_dir = os.path.join(data_dir, "lswsd")
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")

        # fmt: off
        self.UD_POS_TAGS = [
            "NOUN", "PUNCT", "ADP", "NUM", "SYM", "SCONJ", "ADJ", "PART", "DET",
            "CCONJ", "PROPN", "PRON", "X", "_", "ADV", "INTJ", "VERB", "AUX",
        ]
        # https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
        self.penn_to_ud = defaultdict(
            lambda: "_",
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

    def prepare_data(self) -> None:
        """Takes care of downloading data"""
        if self.has_setup:
            return
        if os.path.exists(self.data_dir):
            print("Data already exists. Skipping download.")
            self.raw_dset_dict = datasets.load_from_disk(self.raw_dir)
            return

        print("Downloading SemCor from HuggingFace")
        semcor = datasets.load_dataset("thesofakillers/SemCor")

        print("Performing minor preprocessing")
        # put them all together
        dataset = datasets.concatenate_datasets([semcor[i] for i in semcor])
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
        # these are the lemmas we are interested in
        with open("lswsd_lemmas.txt", "r") as f:
            self.lemmas = set(f.read().splitlines())
        self.id_to_cname = set()
        # processing so that the right columns are available for our classifier
        self.train = self.process_dset(self.raw_dset_dict["train"], split="train")
        self.val = self.process_dset(self.raw_dset_dict["val"], split="val")
        self.test = self.process_dset(self.raw_dset_dict["test"], split="test")

    def process_dset(self, dset: Dataset, split: str) -> Dataset:
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
        print(f"Processing {split} split...")
        processed_path = os.path.join(self.processed_dir, split)

        # don't need to do the remaining processing if we've already done it
        if os.path.exists(processed_path):
            print("Dataset already processed. Loading from disk")
            proc_dataset = datasets.load_from_disk(processed_path)
            if split == "train":
                self.id_to_cname = proc_dataset.features["senses"].feature.names
                self._handle_cname_maps()
            print("Done.")

            return proc_dataset

        dset, whitelisted_sentences = self._build_whitelist(
            dset, True if split == "train" else False
        )

        if split == "train":
            # add "unk" at beginning
            self.id_to_cname = list(self.id_to_cname)
            self._handle_cname_maps()

        filtered_dset = dset.filter(
            self._is_in_whitelisted,
            fn_kwargs={"whitelisted_sentences": whitelisted_sentences},
        )

        filtered_df = filtered_dset.to_pandas()
        filtered_df = filtered_df.replace(to_replace="None", value=None)

        sentence_df = filtered_df.groupby(["tagfile", "pnum", "snum"]).agg(
            tokens=pd.NamedAgg(column="value", aggfunc=list),
            idxs=pd.NamedAgg(column="lemma", aggfunc=self._where_candidates),
            lemmas=pd.NamedAgg(column="lemma", aggfunc=list),
            senses=pd.NamedAgg(column="sense", aggfunc=list),
            pos=pd.NamedAgg(column="pos", aggfunc=list),
        )

        sentence_df = sentence_df.apply(self._filter_cands_map_ids, axis=1)
        sentence_df = sentence_df.reset_index().drop(
            columns=["tagfile", "pnum", "snum"]
        )
        sentence_df["id"] = sentence_df.index

        # converting back into a HF dataset for future use
        proc_dataset = Dataset.from_pandas(
            sentence_df,
            features=Features(
                {
                    "id": Value("int64"),
                    "tokens": Sequence(Value("string")),
                    "lemmas": Sequence(Value("string")),
                    "idxs": Sequence(Value("int64")),
                    "senses": Sequence(ClassLabel(names=list(self.cname_to_id.keys()))),
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

    def _handle_cname_maps(self):
        """Add 'unk' to the beginning of the list of sense names and computes inverse"""
        self.cname_to_id = defaultdict(
            lambda: 0, {sense: i for i, sense in enumerate(self.id_to_cname)}
        )
        self.id_to_cname.insert(0, "unk")

    def _filter_cands_map_ids(self, input_row):
        """Keeps only the lexical samples and maps senses and pos to ids"""
        new_lemmas = []
        new_senses = []
        new_pos = []
        for i in input_row["idxs"]:
            new_lemmas.append(input_row["lemmas"][i])
            new_senses.append(self.cname_to_id[input_row["senses"][i]])
            new_pos.append(self.pos_cname2id[self.penn_to_ud[input_row["pos"][i]]])
        input_row["lemmas"] = new_lemmas
        input_row["senses"] = new_senses
        input_row["pos"] = new_pos
        return input_row

    def _where_candidates(self, agg_input):
        return np.where(agg_input.isin(self.lemmas))[0]

    def _is_in_whitelisted(self, entry, whitelisted_sentences):
        return (
            entry["tagfile"],
            entry["pnum"],
            entry["snum"],
        ) in whitelisted_sentences

    def _build_whitelist(
        self, dset: Dataset, train: bool = False
    ) -> Tuple[Dataset, Set[Tuple[str]]]:
        whitelisted_sentences = set()
        senses = np.empty(len(dset), dtype="str")
        for i, entry in tqdm(enumerate(dset), total=len(dset)):
            sense = str(entry["lemma"]) + "%" + str(entry["lexsn"])
            senses[i] = sense
            if entry["lemma"] in self.lemmas:
                whitelisted_sentences.add(
                    (entry["tagfile"], entry["pnum"], entry["snum"])
                )
                if train:
                    self.id_to_cname.add(sense)
        dset = dset.add_column("sense", senses)
        return dset, whitelisted_sentences

    def get_collate_fn(self):
        if self.hparams.task == "WSD":
            return self.wsd_collate_fn

    def wsd_collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[
        BatchEncoding, List[LongTensor], List[LongTensor], List[LongTensor], List[str]
    ]:
        """
        Parses an incoming batch of data, tokenizing the sentences in the process
        """
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
