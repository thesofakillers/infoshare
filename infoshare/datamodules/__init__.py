import os

from argparse import ArgumentParser

from infoshare.datamodules.base import BaseDataModule
from infoshare.datamodules.ud import UDDataModule
from infoshare.datamodules.wsd import WSDDataModule
from infoshare.datamodules.lswsd import LSWSDDataModule


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
        "WSD",
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
        args.treebank_name,
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
        args.treebank_name,
        tokenize_fn,
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )
    ud_dep.prepare_data()
    ud_dep.setup()

    print("Done.")
