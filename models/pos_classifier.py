from .base_classifier import BaseClassifier
import torch
from torch import nn, Tensor
from transformers import BatchEncoding
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from typing import Tuple


class POSClassifier(BaseClassifier):
    def get_classifier_head(self, n_hidden: int, n_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, encoded_input: BatchEncoding) -> Tensor:
        embeddings = self.bert(encoded_input)
        output = self.classifier(embeddings)
        return embeddings, output

    def process_batch(self, batch: Tuple) -> Tuple[Tensor, Tensor]:
        """given a batch, returns embeddings, logits and targets"""
        encoded_inputs, targets = batch
        embeddings, logits = self(encoded_inputs)
        return embeddings, logits, targets

    def validation_step(self, batch, _):
        batched_embs, batched_preds, targets = self.process_batch(batch)
        batch_size = len(batched_preds)

        acc = self.calculate_average_accuracy(batched_preds, targets)

        # Pad target values to use with CE
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)
        loss = F.cross_entropy(
            batched_preds.permute(0, 2, 1),
            targets_padded,
            ignore_index=-1,
        )

        self.log("val_acc", acc, batch_size=batch_size)
        self.log("val_loss", loss, batch_size=batch_size)

        # for each sentence in batch
        for embs, preds, target in zip(batched_embs, batched_preds, targets):
            preds = preds[: len(target), :].argmax(dim=1).tolist()
            # for each word in sentence
            for emb, pred in zip(embs, preds):
                if pred not in self.tag_centroids:
                    self.tag_centroids[pred] = [emb]
                else:
                    self.tag_centroids[pred].append(emb)

    def on_validation_epoch_start(self):
        # we overwrite tag_centroids every epoch
        self.tag_centroids = {}

    def on_validation_epoch_end(self):
        """
        at the end of the epoch, we take the mean of vectors
        accumulated for each tag and store it
        """
        self.tag_centroids = {
            k: torch.mean(torch.vstack(v), dim=0) for k, v in self.tag_centroids.items()
        }

    def on_save_checkpoint(self, ckpt):
        super().on_save_checkpoint(ckpt)
        ckpt["tag_centroids"] = self.tag_centroids

    def on_load_checkpoint(self, ckpt):
        super().on_load_checkpoint(ckpt)
        self.tag_centroids = ckpt["tag_centroids"]
