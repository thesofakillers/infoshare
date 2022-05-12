from .base_classifier import BaseClassifier
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from typing import List, Tuple

import torch


class DEPClassifier(BaseClassifier):
    def get_classifier_head(self, n_hidden: int, n_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(4 * n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, encoded_input: BatchEncoding, heads: List[Tensor]) -> Tuple[Tensor, Tensor]:
        embeddings = self.bert(encoded_input)

        classifier_input = []
        # Concatenate each word representation with its head's
        for (seq, head) in zip(embeddings, heads):
            seq_nopad = seq[: len(head)]
            parents = seq_nopad[head - 1]
            classifier_input += [
                torch.hstack(
                    [
                        seq_nopad,
                        parents,
                        (seq_nopad - parents).abs(),
                        (seq_nopad * parents),
                    ]
                )
            ]

        classifier_input = pad_sequence(classifier_input, batch_first=True)
        output = self.classifier(classifier_input)
        return embeddings, output

    def process_batch(self, batch: Tuple) -> Tuple[Tensor, Tensor, Tensor]:
        encoded_inputs, heads, targets = batch
        embeddings, logits = self(encoded_inputs, heads)
        return embeddings, logits, targets
