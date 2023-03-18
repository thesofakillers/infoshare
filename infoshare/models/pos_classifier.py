from argparse import ArgumentParser
from typing import Dict, Tuple

from torch import nn, Tensor
from transformers import BatchEncoding

from infoshare.models.base_classifier import BaseClassifier


class POSClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        self.save_hyperparameters()
        super().__init__(**kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("POS Classifier")
        parser.add_argument(
            "--pos_dataset",
            type=str,
            default="ud",
            choices=["ud", "semcor"],
            help="The dataset to use.",
        )
        return parent_parser

    def get_classifier_head(self, n_hidden: int, n_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, encoded_input: BatchEncoding) -> Tuple[Tensor, Tensor]:
        embeddings = self.bert(encoded_input)
        output = self.classifier(embeddings)
        return embeddings, output

    def process_batch(self, batch: Tuple) -> Dict[str, Tensor]:
        encoded_inputs, targets = batch
        embeddings, logits = self(encoded_inputs)
        return {
            "embeddings": embeddings,
            "logits": logits,
            "targets": targets,
        }
