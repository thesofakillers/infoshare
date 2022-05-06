from argparse import ArgumentParser
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, Optimizer
from transformers import BatchEncoding
from typing import List, Tuple

from .bert_encoder import BERTEncoderForWordClassification

import torch
import torch.nn.functional as F
import torchmetrics.functional as TF


class POSClassifier(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("POS Classifier")

        parser.add_argument(
            "--encoder_name",
            type=str,
            default="roberta-base",
            help="The name of the HuggingFace model to use as an encoder.",
        )

        parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            help="The initial learning rate for the classifier.",
        )

        return parent_parser

    def __init__(self, encoder_name: str, n_classes: int, lr: float, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.bert = BERTEncoderForWordClassification(encoder_name)
        n_hidden = self.bert.hidden_size()

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, encoded_input: BatchEncoding) -> Tensor:
        output = self.bert(encoded_input)
        output = self.classifier(output)
        return output

    def step(
        self,
        batch: Tuple[BatchEncoding, List[Tensor]],
        stage: str,
    ) -> Tensor:
        encoded_input, target = batch

        # Call model forward to get logits
        logits = self(encoded_input)
        batch_size = len(logits)

        # Pad target values to use with CE
        target_padded = pad_sequence(target, batch_first=True, padding_value=-1)

        # Calculate CrossEntropy loss
        # NOTE: CE expects input shape (N, C, S) while logits' shape is (N, S, C)
        loss = F.cross_entropy(logits.permute(0, 2, 1), target_padded, ignore_index=-1)
        # loss = F.cross_entropy(logits.permute(0, 2, 1), target_padded, ignore_index=-1, reduction='none').mean()

        with torch.no_grad():
            # Extract sequence length from number of POS tags to predict
            sequence_lengths = [len(_) for _ in target]

            # Calculate accuracy as the mean of all sentences' accuracy
            acc = torch.tensor(
                [
                    TF.accuracy(logits[i, : sequence_lengths[i], :], target[i])
                    for i in range(len(target))
                ]
            ).mean()

        self.log(f"{stage}_acc", acc, batch_size=batch_size)
        self.log(f"{stage}_loss", loss, batch_size=batch_size)

        return loss

    def training_step(
        self,
        batch: Tuple[BatchEncoding, List[Tensor]],
        _: Tensor,
    ) -> Tensor:
        return self.step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[BatchEncoding, List[Tensor]],
        _: Tensor,
    ) -> Tensor:
        return self.step(batch, "val")

    def test_step(
        self,
        batch: Tuple[BatchEncoding, List[Tensor]],
        _: Tensor,
    ) -> Tensor:
        return self.step(batch, "test")

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.classifier.parameters(), lr=self.hparams.lr)

    def on_save_checkpoint(self, checkpoint: dict):
        pass  # TODO: maybe remove BERT from checkpoint?

    def on_load_checkpoint(self, checkpoint: dict):
        pass  # TODO: maybe load BERT dynamically?
