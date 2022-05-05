from argparse import ArgumentParser
from collections import defaultdict
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import AdamW, Optimizer
from transformers import AutoModel, BatchEncoding, BertPreTrainedModel
from typing import List, Tuple
from .bert import BERTForWordClassification

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

        self.bert = BERTForWordClassification(encoder_name)
        n_hidden = self.bert.model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.Tanh(), nn.Linear(n_hidden, n_classes)
        )

    def forward(self, encoded_input: BatchEncoding) -> Tensor:
        output = self.bert(encoded_input)
        output = self.classifier(output)
        
        return output

    def step(
        self,
        batch: Tuple[BatchEncoding, List[Tensor], List[Tensor]],
        stage: str,
    ) -> Tensor:
        encoded_input, token_pos = batch
        
        sequence_lengths = [len(_) for _ in token_pos]

        logits = self(encoded_input)
        
        loss = torch.mean(torch.Tensor(
            [F.cross_entropy(logits[i, :sequence_lengths[i], :], token_pos[i]) 
            for i in range(len(token_pos))]
        ))
        
        # acc = TF.accuracy(logits, flat_pos)

        # self.log(f"{stage}_acc", acc)
        # self.log(f"{stage}_loss", loss)

        return loss

    def training_step(
        self,
        batch: Tuple[List[List[BatchEncoding]], List[Tensor], List[Tensor]],
        _: Tensor,
    ) -> Tensor:
        return self.step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[List[List[BatchEncoding]], List[Tensor], List[Tensor]],
        _: Tensor,
    ) -> Tensor:
        return self.step(batch, "val")

    def test_step(
        self,
        batch: Tuple[List[List[BatchEncoding]], List[Tensor], List[Tensor]],
        _: Tensor,
    ) -> Tensor:
        return self.step(batch, "test")

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.classifier.parameters(), lr=self.hparams.lr)

    def on_save_checkpoint(self, checkpoint: dict):
        pass  # TODO: maybe remove BERT from checkpoint?

    def on_load_checkpoint(self, checkpoint: dict):
        pass  # TODO: maybe load BERT dynamically?
