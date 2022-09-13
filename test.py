from argparse import ArgumentParser, Namespace
from data import UDDataModule
from functools import partial
from models import *
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, logging
from utils import get_experiment_name

import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()


def test(args: Namespace):
    seed_everything(args.seed, workers=True)

    # Load the hyperparameters from checkpoint
    current_device = "cpu" if args.no_gpu or (not torch.cuda.is_available()) else "cuda"
    hparams = torch.load(args.checkpoint, map_location=current_device)["hyper_parameters"]
    hparams = Namespace(**hparams)

    # Load the tokenizer
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

    # Load the model class constructor
    if hparams.task == "DEP":
        model_class = DEPClassifier
    elif hparams.task == "POS":
        model_class = POSClassifier
    else:
        raise Exception(f"Unsupported task: {hparams.task}")

    # Load the BERT encoder
    bert = BERTEncoderForWordClassification(
        encoder_name=hparams.encoder_name,
        aggregation=hparams.aggregation,
        probe_layer=hparams.probe_layer,
    )

    # Load the PL module
    print(f"Loading from checkpoint: {args.checkpoint}")
    model = model_class.load_from_checkpoint(args.checkpoint, num_workers=args.num_workers)
    model.set_encoder(bert)
    if args.centroids_file is not None:
        model.load_centroids(args.centroids_file)

    # Load PL datamodule
    ud = UDDataModule(
        hparams.task,
        hparams.treebank_name,
        tokenize_fn,
        args.data_dir,
        args.batch_size,
        args.num_workers,
    )

    ud.prepare_data()
    ud.setup()

    # Configure the logger
    v_postfix = f"_{args.neutralizer}" if args.neutralizer else ""
    logger = TensorBoardLogger(
        save_dir=os.path.join(
            args.log_dir,
            hparams.encoder_name,
            hparams.treebank_name,
            hparams.task,
        ),
        name=get_experiment_name(hparams),
        version=f"evaluation{v_postfix}",
        default_hp_metric=False,
    )

    # Configure GPU usage
    use_gpu = 0 if args.no_gpu or (not torch.cuda.is_available()) else 1

    # Set up the trainer
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        gpus=use_gpu,
        max_epochs=1,  # just to supress a warning from PL
    )

    if args.neutralizer:
        model.set_neutralizer(args.neutralizer)

    # Test the model
    trainer.test(model, ud)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The checkpoint from which to load a model.",
    )

    parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        help="Whether to enable the progress bar (NOT recommended when logging to file).",
    )

    parser.add_argument(
        "--neutralizer",
        type=str,
        help="The target class to (cross-)neutralize all embeddings with.",
    )

    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Whether to NOT use a GPU accelerator for training.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=420,
        help="The seed to use for the RNG.",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./lightning_logs",
        help="The logging directory for Pytorch Lightning.",
    )

    # Dataset arguments
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
        help="The data directory to use for the datasets.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The number of subprocesses used by the dataloaders.",
    )
    parser.add_argument(
        "--centroids_file",
        type=str,
        default=None,
        help="Custom centroids file to use for neutralisation. (Optional)",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    test(args)
