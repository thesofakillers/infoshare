from .bert_encoder import BERTEncoderForWordClassification
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from functools import partial
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, Optimizer
from transformers import AutoTokenizer
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchmetrics.functional as TF


class BaseClassifier(LightningModule, metaclass=ABCMeta):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("Probing Model")
        parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="The initial learning rate for the probing classifier.",
        )
        return parent_parser

    def __init__(
        self,
        n_hidden: int,
        n_classes: int,
        class_map: Dict[int, str],
        lr: float = 1e-3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.classifier = self.get_classifier_head(n_hidden, n_classes)

    def set_encoder(self, bert: BERTEncoderForWordClassification):
        self.bert = bert

        tokenizer = AutoTokenizer.from_pretrained(
            bert.encoder_name,
            add_prefix_space=True,
        )
        self.tokenize_fn = partial(tokenizer, return_tensors="pt")

    @abstractmethod
    def get_classifier_head(self, n_hidden, n_classes: int) -> nn.Module:
        raise NotImplementedError()

    @torch.no_grad()
    def infer(self, sentence: str) -> List[str]:
        # Encode sentence
        encoding = self.tokenize_fn(sentence)
        # Predict a class for each word
        output = self(encoding).squeeze(dim=0)
        # Map each prediction to its class name
        output = [self.hparams.class_map[i] for i in output.argmax(dim=1)]
        return output

    @abstractmethod
    def get_logits_targets(self, batch: Tuple) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def step(self, batch: Tuple, stage: str) -> Tensor:
        logits, targets = self.get_logits_targets(batch)
        batch_size = len(logits)

        # Pad target values to use with CE
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)

        # Calculate CrossEntropy loss
        # NOTE: CE expects input shape (N, C, S) while logits' shape is (N, S, C)
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets_padded, ignore_index=-1)
        # loss = F.cross_entropy(logits.permute(0, 2, 1), targets_padded, ignore_index=-1, reduction='none').mean()

        with torch.no_grad():
            # Extract sequence length from number of POS tags to predict
            sequence_lengths = [len(_) for _ in targets]

            # Calculate accuracy as the mean of all sentences' accuracy
            acc = torch.tensor(
                [
                    TF.accuracy(logits[i, : sequence_lengths[i], :], targets[i])
                    for i in range(len(targets))
                ]
            ).mean()

        self.log(f"{stage}_acc", acc, batch_size=batch_size)
        self.log(f"{stage}_loss", loss, batch_size=batch_size)

        return loss

    def training_step(self, batch: Tuple, _: Tensor) -> Tensor:
        return self.step(batch, "train")

    def validation_step(self, batch: Tuple, _: Tensor) -> Tensor:
        return self.step(batch, "val")

    def test_step(self, batch: Tuple, _: Tensor) -> Tensor:
        return self.step(batch, "test")

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.classifier.parameters(), lr=self.hparams.lr)

    def on_save_checkpoint(self, ckpt: dict):
        # Remove BERT from checkpoint as we can load it dynamically
        keys = list(ckpt["state_dict"].keys())
        for key in keys:
            if key.startswith("bert."):
                del ckpt["state_dict"][key]
