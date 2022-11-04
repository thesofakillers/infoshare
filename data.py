import os
from typing import Any, Callable, Dict, List, Tuple, Optional
from argparse import ArgumentParser
from collections import defaultdict
import xml.etree.ElementTree as ET

from datasets import load_dataset, load_from_disk, Features, Sequence, Value, ClassLabel
from datasets.arrow_dataset import Dataset
from pytorch_lightning import LightningDataModule
from torch import LongTensor
from torch.utils.data import DataLoader
from transformers import BatchEncoding
import pandas as pd
import numpy as np

from utils import download_and_unzip


class BaseDataModule(LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Data")
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
            choices=["DEP", "POS", "WSD"],
            help="The task to train the probing classifier on.",
        )


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
            keep_indices = [idx for idx, value in enumerate(x["head"]) if value != "None"]
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
                self.num_classes = self.ud_train.info.features["upos"].feature.num_classes
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

    def pos_collate_fn(self, batch: List[Dict[str, Any]]) -> Tuple[BatchEncoding, List[LongTensor]]:
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
    lemma_to_sense_ids Dict[str, List[int]]
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
        self.cname_to_id = defaultdict(lambda: 0, self.cname_to_id)
        gold_labels["sense_id"] = gold_labels.label.map(self.cname_to_id)
        gold_labels["lemma"] = gold_labels["label"].str.split("%").str[0]
        self.lemma_to_sense_ids = gold_labels.groupby("lemma")["sense_id"].apply(list).to_dict()
        self.lemma_to_sense_ids = defaultdict(lambda: [0], self.lemma_to_sense_ids)

    def wsd_dset(self, dataset_path: str, is_train: bool = False) -> Dataset:
        """
        Given path to Raganato benchmark XML and gold labels
        Returns a torch dataset or hf dataset
        """
        processed_path = os.path.join(dataset_path, "processed")
        dataset_name = dataset_path.split("/")[-1]
        print(f"Creating dataset for {dataset_name}")

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
            print("Dataset already processed. Loading from disk")
            dataset = load_from_disk(processed_path)

            return dataset
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
                        senses.append(self.cname_to_id[gold_labels["label"].loc[word.get("id")]])
                data.append((sentence.get("id"), sent_words, sent_lemmas, idxs, senses, pos))
        # convert to dataframe
        data_df = pd.DataFrame(data, columns=["id", "tokens", "lemmas", "idxs", "senses", "pos"])
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
                        ClassLabel(num_classes=len(self.POS_TAGS), names=self.pos_id2cname)
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


if __name__ == "__main__":
    # if for whatever reason you want to setup data separately
    from functools import partial
    from transformers import AutoTokenizer

    parser = ArgumentParser("Setup data beforehand")
    WSDDataModule.add_model_specific_args(parser)
    UDDataModule.add_model_specific_args(parser)
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
    print("Setting up WSD")
    wsd = WSDDataModule(
        args.task,
        tokenize_fn,
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )
    wsd.prepare_data()
    wsd.setup()

    print("Setting up UD pos")
    ud_pos = UDDataModule(
        "POS",
        tokenize_fn,
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )
    ud_pos.prepare_data()
    ud_pos.setup()

    print("Setting up UD DEP")
    ud_dep = UDDataModule(
        "DEP",
        tokenize_fn,
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )
    ud_dep.prepare_data()
    ud_dep.setup()

    print("Done.")
