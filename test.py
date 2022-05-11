from argparse import ArgumentParser, Namespace
from data import UDDataModule
from functools import partial
from models import *
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, logging

import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()


def test(args: Namespace):
    seed_everything(args.seed, workers=True)

    # load hparams from checkpoint
    hparams = torch.load(args.checkpoint)["hyper_parameters"]

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.encoder_name,
        add_prefix_space=True,
    )
    tokenize_fn = partial(
        tokenizer,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
    )

    # load model class constructor
    if hparams.task == "DEP":
        model_class = DEPClassifier
    elif hparams.task == "POS":
        model_class = POSClassifier
    else:
        raise Exception(f"Unsupported task: {hparams.task}")

    # load BERT encoder
    bert = BERTEncoderForWordClassification(
        encoder_name=hparams.encoder_name,
        aggregation=hparams.aggregation,
        probe_layer=hparams.probe_layer,
    )

    # load PL module
    print(f"Loading from checkpoint: {args.checkpoint}")
    model = model_class.load_from_checkpoint(args.checkpoint)
    model.set_encoder(bert)

    # load PL datamodule
    ud = UDDataModule(
        hparams.task,
        hparams.treebank_name,
        tokenize_fn,
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )

    ud.prepare_data()
    ud.setup("test")

    # configure logger
    logger = TensorBoardLogger(
        args.log_dir,
        name=hparams.encoder_name,
        default_hp_metric=False,
    )

    # set up trainer
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        gpus=(0 if args.no_gpu else 1),
    )

    # test the model
    trainer.test(model, ud)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=420,
        help="The seed to use for the RNG.",
    )

    parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        default=True,  # TODO: remove this before running on lisa
        help="Whether to enable the progress bar (NOT recommended when logging to file).",
    )

    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Whether to NOT use a GPU accelerator for training.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The checkpoint from which to load a model.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size used by the dataloaders.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of subprocesses used by the dataloaders.",
    )

    # Directory arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="The data directory to use for the datasets.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./lightning_logs",
        help="The logging directory for Pytorch Lightning.",
    )

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    test(args)
