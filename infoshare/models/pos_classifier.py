from typing import Dict, Tuple

from torch import nn, Tensor
from transformers import BatchEncoding

from infoshare.models.base_classifier import BaseClassifier


class POSClassifier(BaseClassifier):
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
