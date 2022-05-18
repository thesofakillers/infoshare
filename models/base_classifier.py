from .bert_encoder import BERTEncoderForWordClassification
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, Optimizer
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple

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
        class_map: List[str],
        lr: float = 1e-3,
        ignore_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # maps target human-readable labels to their IDs
        self.label_to_id = {label: i for i, label in enumerate(self.hparams.class_map)}

        self.classifier = self.get_classifier_head(n_hidden, n_classes)

        # no neutralizer unless set with set_neutralizer(str)
        self.has_neutralizer = False

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
    def process_batch(self, batch: Tuple) -> Dict[str, Tensor]:
        raise NotImplementedError()

    def intercept_embeddings(self, **kwargs) -> Tensor:
        return kwargs["embeddings"]

    @torch.no_grad()
    def calculate_average_accuracy(
        self,
        logits: Tensor,
        targets: Tensor,
        average: str = "micro",
    ) -> Tensor:
        # Extract sequence length from number of POS tags to predict
        sequence_lengths = [len(_) for _ in targets]

        # Calculate accuracy as the mean over all sentences
        acc = torch.vstack(
            [
                TF.accuracy(
                    preds=logits[i, : sequence_lengths[i], :],
                    target=targets[i],
                    average=average,
                    num_classes=logits.shape[-1],
                    ignore_index=self.hparams.ignore_idx,
                )
                for i in range(len(targets))
            ]
        ).nanmean(dim=0)

        return acc

    def on_validation_epoch_start(self):
        # we overwrite class_centroids at every epoch
        self.class_centroids = defaultdict(list)

    def on_validation_epoch_end(self):
        """
        at the end of the epoch, we take the mean of vectors
        accumulated for each class and store it
        """
        self.class_centroids = {
            k: torch.mean(torch.vstack(v), dim=0) for k, v in self.class_centroids.items()
        }

    def postprocess_val_batch(self, batch_embs: Tensor, batch_logits: Tensor, targets: Tensor):
        for embs, logits, target in zip(batch_embs, batch_logits, targets):
            preds = logits[: len(target), :].argmax(dim=1).tolist()
            # for each word in sentence
            for emb, pred in zip(embs, preds):
                self.class_centroids[pred] += [emb]

    def fit_step(self, batch: Tuple, stage: str) -> Tensor:
        processed_batch = self.process_batch(batch)
        batch_embs = processed_batch["embeddings"]
        batch_logits = processed_batch["logits"]
        targets = processed_batch["targets"]
        batch_size = len(batch_logits)

        # Pad & mask target values to use with CE
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)
        targets_padded[targets_padded == self.hparams.ignore_idx] = -1

        # Calculate & log CrossEntropy loss
        # NOTE: CE expects input shape (N, C, S) while logits' shape is (N, S, C)
        loss = F.cross_entropy(batch_logits.permute(0, 2, 1), targets_padded, ignore_index=-1)
        # loss = F.cross_entropy(batch_logits.permute(0, 2, 1), targets_padded, ignore_index=-1, reduction='none').mean()
        self.log(f"{stage}_loss", loss, batch_size=batch_size)

        # Calculate & log average accuracy
        self.log_accuracy(batch_logits, targets, stage=stage)

        # Postprocess batch to save tag centroids
        if stage == "val":
            self.postprocess_val_batch(batch_embs, batch_logits, targets)

        return loss

    def training_step(self, batch: Tuple, _: Tensor) -> Tensor:
        return self.fit_step(batch, "train")

    def validation_step(self, batch: Tuple, _: Tensor):
        self.fit_step(batch, "val")

    def on_test_start(self):
        # Move centroids to model's device
        for class_id, centroid in self.class_centroids.items():
            self.class_centroids[class_id] = centroid.to(self.device)

    def test_step(self, batch: Tuple, _: Tensor):
        processed_batch = self.process_batch(batch)
        embs = processed_batch["embeddings"]
        logits = processed_batch["logits"]
        targets = processed_batch["targets"]

        if not self.has_neutralizer:
            # perform standard testing
            self.log_accuracy(logits, targets, stage="test")
        else:
            # perform the testing on neutralized embeddings
            neutral_embs = self.subtract_centroid(embs)
            processed_batch["embeddings"] = neutral_embs
            neutral_embs = self.intercept_embeddings(**processed_batch)
            neutral_logits = self.classifier(neutral_embs)
            self.log_accuracy(
                neutral_logits,
                targets,
                stage="test",
                prefix=f"{self.neutralizer}/",
            )

    def subtract_centroid(self, embs: Tensor) -> Tensor:
        """
        Subtracts the centroid of some neutralizing label from a set of embeddings
        """
        neutralizer_id = self.label_to_id[self.neutralizer]
        centroid_emb = self.class_centroids[neutralizer_id]
        neutral_embs = embs - centroid_emb
        return neutral_embs

    def set_neutralizer(self, neutralizer_str: str):
        assert (
            neutralizer_str in self.label_to_id.keys()
        ), f"Invalid neutralizer string used, must be one of {self.label_to_id.keys()}"
        self.neutralizer = neutralizer_str
        self.has_neutralizer = True

    def log_accuracy(self, logits: Tensor, targets: Tensor, stage: str, prefix: str = ""):
        batch_size = len(logits)

        # Calculate & log average micro accuracy
        acc_avg = self.calculate_average_accuracy(logits, targets)
        self.log(f"{stage}_acc", acc_avg, batch_size=batch_size)

        if stage != "test":
            # No need to log per-class accuracy for train & val
            return

        # Calculate & log average accuracy per-class
        acc_per_class = self.calculate_average_accuracy(logits, targets, average="none")
        for i, acc_i in enumerate(acc_per_class):
            class_name = self.hparams.class_map[i]
            self.log(f"{prefix}{stage}_acc_{class_name}", acc_i, batch_size=batch_size)

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.classifier.parameters(), lr=self.hparams.lr)

    def on_save_checkpoint(self, ckpt: dict):
        # Save class centroids
        ckpt["class_centroids"] = self.class_centroids

        # Remove BERT from checkpoint as we can load it dynamically
        keys = list(ckpt["state_dict"].keys())
        for key in keys:
            if key.startswith("bert."):
                del ckpt["state_dict"][key]

    def on_load_checkpoint(self, ckpt: dict):
        # Load class centroids
        self.class_centroids = ckpt["class_centroids"]
