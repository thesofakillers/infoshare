from argparse import ArgumentParser
from .base_classifier import BaseClassifier
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from typing import Dict, List, Tuple

import torch


class DEPClassifier(BaseClassifier):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("DEP Classifier")
        parser.add_argument(
            "--concat_mode",
            type=str,
            default="ABS",
            choices=["ABS", "MEAN", "ONLY"],
            help="How to concatenated child and parent.",
        )
        return parent_parser

    def get_classifier_head(self, n_hidden: int, n_classes: int) -> nn.Module:
        input_multiplier = 2  # 2x input for "ONLY" concat_mode
        if self.hparams.concat_mode in {"ABS", "MEAN"}:
            input_multiplier += 2  # 4x input for "ABS" and "MEAN" concat_modes
        return nn.Sequential(
            nn.Linear(input_multiplier * n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_classes),
        )

    def intercept_embeddings(self, **kwargs) -> Tensor:
        embeddings = kwargs["embeddings"]
        heads = kwargs["heads"]

        classifier_input = []
        # Concatenate each word representation with its head's
        for (seq, head) in zip(embeddings, heads):
            seq_nopad = seq[: len(head)]
            parents = seq_nopad[head - 1]
            # handles base case where self.hparams.concat_mode == 'ONLY'
            tensors_to_stack = [seq_nopad, parents]
            # in ABS and MEAN cases, we concatenate additional representations
            if self.hparams.concat_mode == "ABS":
                tensors_to_stack += [(seq_nopad - parents).abs(), (seq_nopad * parents)]
            elif self.hparams.concat_mode == "MEAN":
                tensors_to_stack += [
                    torch.mean(torch.stack([seq_nopad, parents]), dim=0),
                    (seq_nopad * parents),
                ]
            classifier_input += [torch.hstack(tensors_to_stack)]

        classifier_input = pad_sequence(classifier_input, batch_first=True)
        return classifier_input

    def forward(self, encoded_input: BatchEncoding, heads: List[Tensor]) -> Tuple[Tensor, Tensor]:
        embeddings = self.bert(encoded_input)
        concatenated_embs = self.intercept_embeddings(embeddings=embeddings, heads=heads)
        output = self.classifier(concatenated_embs)
        return concatenated_embs, output

    def process_batch(self, batch: Tuple) -> Dict[str, Tensor]:
        encoded_inputs, heads, targets = batch
        embeddings, logits = self(encoded_inputs, heads)
        return {
            "embeddings": embeddings,
            "heads": heads,
            "logits": logits,
            "targets": targets,
        }
