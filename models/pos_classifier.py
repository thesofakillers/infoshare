from .base_classifier import BaseClassifier
from torch import nn, Tensor
from transformers import BatchEncoding
from typing import Tuple


class POSClassifier(BaseClassifier):
    def get_classifier_head(self, n_hidden: int, n_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, encoded_input: BatchEncoding) -> Tensor:
        output = self.bert(encoded_input)
        output = self.classifier(output)
        return output

    def get_logits_targets(self, batch: Tuple) -> Tuple[Tensor, Tensor]:
        encoded_inputs, targets = batch
        logits = self(encoded_inputs)
        return logits, targets
