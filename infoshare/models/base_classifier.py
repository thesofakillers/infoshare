from typing import Callable, Dict, List, Optional, Tuple, Union
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial

from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, Optimizer
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import torchmetrics.functional as TF

from infoshare.models.bert_encoder import BERTEncoderForWordClassification


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

    # Declare variables that will be initialized later
    bert: BERTEncoderForWordClassification
    tokenize_fn: Callable
    class_centroids: Dict[int, Union[Tensor, List[Tensor]]]
    neutralizer: Optional[str]

    def __init__(
        self,
        n_hidden: int,
        n_classes: int,
        class_map: List[str],
        lr: float = 1e-3,
        ignore_id: Optional[int] = None,
        compute_centroids: bool = True,
        **kwargs,
    ):
        """Abstract classifier class for probing tasks.

        Note: most specific subclass should call self.save_hyperparameters()
        in their __init__.

        Args:
            n_hidden (int): the number of hidden units in the classifier head
            n_classes (int): the number of target classes
            class_map (List[str]): the mapping of class ids to class names
            lr (float): the learning rate for the classifier
            ignore_id (int): the id to ignore in the target vector
            compute_centroids (bool): whether to compute class centroids. Default True.
        """
        super().__init__()

        # maps target human-readable labels to their IDs
        self.label_to_id = {label: i for i, label in enumerate(self.hparams.class_map)}

        self.classifier = self.get_classifier_head(n_hidden, n_classes)

        # no neutralizer unless set with set_neutralizer(str)
        self.has_neutralizer = False

    ###########
    # Setters #
    ###########

    def set_encoder(self, bert: BERTEncoderForWordClassification):
        """Set the BERT encoder to use for probing tasks.
        This is handled outside the constructor to deal with loading from checkpoints.
        """
        self.bert = bert

        tokenizer = AutoTokenizer.from_pretrained(
            bert.encoder_name,
            add_prefix_space=True,
        )
        self.tokenize_fn = partial(tokenizer, return_tensors="pt")

    def set_neutralizer(self, neutralizer_str: str):
        """Sets the neutralizer class to use for the evaluation."""
        assert (
            neutralizer_str in self.label_to_id.keys()
        ), f"Invalid neutralizer string used, must be one of {self.label_to_id.keys()}"
        self.neutralizer = neutralizer_str
        self.has_neutralizer = True

    ####################
    # Abstract methods #
    ####################

    @abstractmethod
    def get_classifier_head(self, n_hidden: int, n_classes: int) -> nn.Module:
        """Defines the torch module used as the classifier head."""
        raise NotImplementedError()

    @abstractmethod
    def process_batch(self, batch: Tuple) -> Dict[str, Tensor]:
        """Handles the unpacking of the batch into a dictionary of tensors."""
        raise NotImplementedError()

    def intercept_embeddings(self, **kwargs) -> Tensor:
        """Intercepts the embeddings from the BERT encoder when needed (e.g. DEP representations)."""
        return kwargs["embeddings"]

    #########################
    # Training & validation #
    #########################

    def training_step(self, batch: Tuple, _: Tensor) -> Tensor:
        return self.fit_step(batch, "train")

    def on_validation_epoch_start(self):
        # We overwrite class_centroids at every epoch
        self.class_centroids = defaultdict(list)

    def validation_step(self, batch: Tuple, _: Tensor):
        self.fit_step(batch, "val")

    def on_validation_epoch_end(self):
        # At the end of the epoch, we take the mean of vectors
        # accumulated for each class and store it in class_centroids
        self.class_centroids = {
            k: torch.mean(torch.vstack(v), dim=0)
            for k, v in self.class_centroids.items()
        }

    def fit_step(self, batch: Tuple, stage: str) -> Tensor:
        processed_batch = self.process_batch(batch)
        batch_embs = processed_batch["embeddings"]
        batch_logits = processed_batch["logits"]
        targets = processed_batch["targets"]
        batch_size = len(batch_logits)

        # Pad & mask target values to use with CE
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)
        targets_padded[targets_padded == self.hparams.ignore_id] = -1

        # Calculate & log CrossEntropy loss
        # NOTE: CE expects input shape (N, C, S) while logits' shape is (N, S, C)
        loss = F.cross_entropy(
            batch_logits.permute(0, 2, 1), targets_padded, ignore_index=-1
        )
        self.log(f"{stage}_loss", loss, batch_size=batch_size)

        # Calculate & log average accuracy
        self.log_metrics(processed_batch, stage)

        # Postprocess batch to save tag centroids (if we want them)
        if stage == "val" and self.hparams.compute_centroids:
            self.postprocess_val_batch(batch_embs, batch_logits, targets)

        return loss

    def postprocess_val_batch(
        self, batch_embs: Tensor, batch_logits: Tensor, targets: Tensor
    ):
        """Postprocess the validation batch in order to calculate the class centroids."""
        for embs, logits, target in zip(batch_embs, batch_logits, targets):
            preds = logits[: len(target), :].argmax(dim=1).tolist()
            # for each word in sentence
            for emb, pred in zip(embs, preds):
                self.class_centroids[pred] += [emb]

    ##############
    # Evaluation #
    ##############

    def on_test_start(self):
        # Move centroids to model's device
        for class_id, centroid in self.class_centroids.items():
            self.class_centroids[class_id] = centroid.to(self.device)

    def test_step(
        self,
        batch: Tuple,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ):
        processed_batch = self.process_batch(batch)

        if not self.has_neutralizer:
            # Perform standard testing
            self.log_metrics(
                processed_batch, stage="test", prefix="", dloader_idx=dataloader_idx
            )
        else:
            # Perform the testing on neutralized embeddings
            neutral_embs = self.subtract_centroid(processed_batch["embeddings"])
            processed_batch["embeddings"] = neutral_embs
            processed_batch["logits"] = self.classifier(neutral_embs)
            self.log_metrics(
                processed_batch,
                stage="test",
                prefix=f"{self.neutralizer}/",
                dloader_idx=dataloader_idx,
            )

    def subtract_centroid(self, embs: Tensor) -> Tensor:
        """
        Subtracts the centroid of some neutralizing label from a set of embeddings
        """
        neutralizer_id = self.label_to_id[self.neutralizer]
        centroid_emb = self.class_centroids[neutralizer_id]
        # check for length mismatch and 0-pad or truncate (happens in x-task x-neutral)
        if embs.shape[-1] != centroid_emb.shape[0]:
            if centroid_emb.shape[0] < embs.shape[-1]:
                centroid_emb = torch.cat(
                    [
                        centroid_emb,
                        torch.zeros(
                            embs.shape[-1] - centroid_emb.shape[0], device=self.device
                        ),
                    ]
                )
            elif centroid_emb.shape[0] > embs.shape[-1]:
                centroid_emb = centroid_emb[: embs.shape[-1]]
        neutral_embs = embs - centroid_emb
        return neutral_embs

    ####################
    # Module optimizer #
    ####################

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.classifier.parameters(), lr=self.hparams.lr)

    ###############
    # Checkpoints #
    ###############

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

    #####################
    # Logging & metrics #
    #####################

    def log_metrics(
        self,
        processed_batch: Dict,
        stage: str,
        prefix: str = "",
        dloader_idx: Optional[int] = None,
    ):
        batch_logits = processed_batch["logits"]
        targets = processed_batch["targets"]
        self.log_accuracy(batch_logits, targets, stage=stage, prefix=prefix)

    def log_accuracy(
        self, logits: Tensor, targets: Tensor, stage: str, prefix: str = ""
    ):
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

    @torch.no_grad()
    def calculate_average_accuracy(
        self,
        logits: Tensor,
        targets: Tensor,
        average: str = "micro",
    ) -> Tensor:
        """Calculates the mean accuracy based on the averaging method specified."""
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
                    ignore_index=self.hparams.ignore_id,
                )
                for i in range(len(targets))
            ]
        ).nanmean(dim=0)

        return acc

    @torch.no_grad()
    def load_centroids(self, file_path: str):
        # load centroids from file
        centroids = torch.load(file_path)
        # override label_to_id
        self.label_to_id = {
            label: i for i, label in enumerate(sorted(centroids.keys()))
        }
        # override class_centroids, consistently with label_to_id
        self.class_centroids = {
            i: centroids[label].to(self.device)
            for i, label in enumerate(sorted(centroids.keys()))
        }

    #######################
    # Model demonstration #
    #######################

    @torch.no_grad()
    def infer(self, sentence: str) -> List[str]:
        """Infer the labels for a given sentence."""
        # Encode sentence
        encoding = self.tokenize_fn(sentence)
        # Predict a class for each word
        _, output = self(encoding)
        output = output.squeeze(dim=0)
        # Map each prediction to its class name
        output = [self.hparams.class_map[i] for i in output.argmax(dim=1)]
        return output
