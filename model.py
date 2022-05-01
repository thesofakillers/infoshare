from argparse import ArgumentParser
from collections import defaultdict
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import AdamW, Optimizer
from transformers import AutoModel, BatchEncoding, BertPreTrainedModel
from typing import List, Tuple

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

        self.bert = self._load_frozen_model(encoder_name)
        n_hidden = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden, n_hidden), nn.Tanh(), nn.Linear(n_hidden, n_classes)
        )

    @staticmethod
    def _load_frozen_model(name: str) -> BertPreTrainedModel:
        bert = AutoModel.from_pretrained(name)

        # freeze BERT's parameters
        for param in bert.parameters():
            param.requires_grad = False

        return bert

    def forward(
        self, encoded_input: BatchEncoding, word_positions: List[Tensor]
    ) -> Tensor:
        batch_output = []

        bert_output = self.bert(**encoded_input).last_hidden_state
        for sample_out, wpos_map in zip(bert_output, word_positions):
            # initialize dict for the tensors corresponding to each word
            word_tensors = defaultdict(list)
            for tensor, token_id in zip(sample_out, wpos_map):
                word_tensors[token_id.item()] += [tensor]

            # remove redundant tensors for special tokens
            del word_tensors[-1]

            # average embeddings per word
            avg_tensors = {
                t_id: torch.vstack(w_t).mean(dim=0)
                for t_id, w_t in word_tensors.items()
            }

            # order tensors by their corresponding word id
            max_word_id = max(wpos_map)
            ordered_tensors = torch.vstack(
                [avg_tensors[i] for i in range(max_word_id + 1)]
            )

            # somehow re-batch items
            batch_output += ordered_tensors

        # vstack output
        batch_output = torch.vstack(batch_output)
        return self.classifier(batch_output)

    def step(
        self,
        batch: Tuple[List[List[BatchEncoding]], List[Tensor], List[Tensor]],
        stage: str,
    ) -> Tensor:
        encodings, token_pos, wordpiece_map = batch
        flat_pos = torch.hstack(token_pos)

        logits = self(encodings, wordpiece_map)
        loss = F.cross_entropy(logits, flat_pos)
        acc = TF.accuracy(logits, flat_pos)

        self.log(f"{stage}_acc", acc, batch_size=len(flat_pos))
        self.log(f"{stage}_loss", loss, batch_size=len(flat_pos))

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
