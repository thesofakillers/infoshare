import os

from argparse import ArgumentParser, Namespace
from functools import partial
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, logging
import torch

from infoshare.utils import get_experiment_name
import infoshare.models as models
from infoshare.datamodules import (
    UDDataModule,
    WSDDataModule,
    BaseDataModule,
    LSWSDDataModule,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()


def train(args: Namespace):
    seed_everything(args.seed, workers=True)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, add_prefix_space=True)
    tokenize_fn = partial(
        tokenizer, is_split_into_words=True, return_tensors="pt", padding=True
    )

    # Load the appropriate datamodule
    if args.task in {"POS", "DEP"}:
        metric = "acc"
        datamodule = UDDataModule(
            args.task,
            args.treebank_name,
            tokenize_fn,
            args.data_dir,
            args.batch_size,
            args.num_workers,
        )
        log_save_dir = os.path.join(
            args.log_dir,
            args.encoder_name,
            args.treebank_name,
            args.task,
        )
    elif args.task in {"WSD"}:
        metric = "acc"
        datamodule = WSDDataModule(
            args.task, tokenize_fn, args.data_dir, args.batch_size, args.num_workers
        )
        log_save_dir = os.path.join(args.log_dir, args.encoder_name, args.task)
    elif args.task in {"LSWSD"}:
        metric = "acc"
        datamodule = LSWSDDataModule(
            args.task, tokenize_fn, args.data_dir, args.batch_size, args.num_workers
        )
        log_save_dir = os.path.join(args.log_dir, args.encoder_name, args.task)
    else:
        raise ValueError(f"Unknown task {args.task}")

    datamodule.prepare_data()
    datamodule.setup("fit")

    # Load the model class constructor
    if args.task == "DEP":
        model_class = models.DEPClassifier
    elif args.task == "POS":
        model_class = models.POSClassifier
    elif args.task in {"WSD", "LSWSD"}:
        if args.task == "WSD":
            model_class = models.WSDClassifier
        else:
            model_class = models.LSWSDClassifier
        # additional args necessary
        args.pos_map = datamodule.pos_id2cname
        args.lemma_to_sense_ids = datamodule.lemma_to_sense_ids
        args.compute_centroids = False
    else:
        raise Exception(f"Unsupported task: {args.task}")

    # Load the BERT encoder
    bert = models.BERTEncoderForWordClassification(**vars(args))

    # Load the PL module
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        model = model_class.load_from_checkpoint(args.checkpoint)
    else:
        model_args = {
            "n_hidden": bert.hidden_size,
            "n_classes": datamodule.num_classes,
            "class_map": datamodule.id_to_cname,
            **vars(args),
        }

        # Ignore "root" predictions for the loss/accuracy in the DEP task
        if args.task == "DEP":
            model_args["ignore_id"] = datamodule.cname_to_id["root"]

        model = model_class(**model_args)

    model.set_encoder(bert)

    # configure logger
    logger = TensorBoardLogger(
        save_dir=log_save_dir,
        name=get_experiment_name(args),
        default_hp_metric=False,
    )

    # Configure the callbacks
    callback_cfg = {"monitor": f"val_{metric}", "mode": "max"}
    es_cb = EarlyStopping(
        **callback_cfg
    )  # TODO: maybe setup other early stopping parameters
    ckpt_cb = ModelCheckpoint(save_top_k=1, **callback_cfg)

    # Configure GPU usage
    use_gpu = 0 if args.no_gpu or (not torch.cuda.is_available()) else 1

    # Set up the trainer
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        gpus=use_gpu,
        callbacks=[es_cb, ckpt_cb],
    )

    trainer_args = {}
    if args.checkpoint:
        trainer_args["ckpt_path"] = args.checkpoint

    # Fit the model
    trainer.fit(model, datamodule, **trainer_args)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument(
        "--checkpoint", type=str, help="The checkpoint from which to load a model."
    )

    parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        help="Whether to enable the progress bar (NOT recommended when logging to file).",
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="./lightning_logs",
        help="The logging directory for Pytorch Lightning.",
    )

    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="The number of steps (batches) between logging to tensorboard.",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="The max amount of epochs to train the classifier.",
    )

    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Whether to NOT use a GPU accelerator for training.",
    )

    parser.add_argument(
        "--seed", type=int, default=420, help="The seed to use for the RNG."
    )

    # Encoder arguments
    models.BERTEncoderForWordClassification.add_model_specific_args(parser)

    # Classifier arguments. POS and WSD, LSWSD covered by Base. DEP has additional args.
    models.BaseClassifier.add_model_specific_args(parser)
    models.DEPClassifier.add_model_specific_args(parser)

    # Dataset arguments. UD has additional args.
    BaseDataModule.add_model_specific_args(parser)
    UDDataModule.add_model_specific_args(parser)

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train(args)
