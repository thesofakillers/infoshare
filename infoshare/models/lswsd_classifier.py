from typing import Tuple, Dict, Union, List, Callable, Optional

import torch.nn as nn
from torch import LongTensor, Tensor
import torch
from transformers import BatchEncoding
from torch.nn.utils.rnn import pad_sequence
import torchmetrics.functional as TF

from infoshare.models.base_classifier import BaseClassifier


class LSWSDClassifier(BaseClassifier):
    """
    Same as POS Classifier but in forward we need to index the
    embeddings with salient idxs, since we no longer have a label for every word.

    We also slightly modify the logging so that we also log F1 score, besides acc
    """

    def get_classifier_head(self, n_hidden: int, n_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(
        self,
        encoded_input: BatchEncoding,
        salient_idxs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        embeddings = self.bert(encoded_input)
        # need to index the (padded) embeddings with salient idxs
        embeddings = [emb[idx] for idx, emb in zip(salient_idxs, embeddings)]
        # indexing breaks padding (max_sequence_len -> max_salient_len), so pad again
        embeddings = pad_sequence(embeddings, batch_first=True)
        logits = self.classifier(embeddings)
        return embeddings, logits

    def process_batch(self, batch: Tuple) -> Dict[str, Union[Tensor, List]]:
        encodings, targets, idxs, pos, lemmas = batch
        embeddings, logits = self(encodings, idxs)
        return {
            "embeddings": embeddings,
            "logits": logits,
            "targets": targets,
            "pos": pos,
            "lemmas": lemmas,
        }

    def log_metrics(
        self,
        processed_batch: Dict,
        stage: str,
        prefix: str = "",
        dloader_idx: Optional[int] = None,
    ):
        batch_logits = processed_batch["logits"]
        batch_targets = processed_batch["targets"]
        batch_lemmas = processed_batch["lemmas"]

        self.log_metric(
            batch_logits,
            batch_targets,
            batch_lemmas,
            stage=stage,
            prefix=prefix,
            metric_name="acc",
        )
        self.log_metric(
            batch_logits,
            batch_targets,
            batch_lemmas,
            stage=stage,
            prefix=prefix,
            metric_name="f1",
        )

    def log_metric(
        self,
        logits: Tensor,
        targets: List[List[LongTensor]],
        lemmas: List[List[str]],
        stage: str,
        prefix: str = "",
        metric_name: str = "acc",
    ):
        batch_size = len(logits)

        # Calculate & log average micro accuracy
        acc_avg = self.calculate_average_metric(logits, targets, lemmas, metric_name)
        self.log(f"{stage}_acc", acc_avg, batch_size=batch_size)

        if stage != "test":
            # No need to log per-class accuracy for train & val
            return

        # Calculate & log average accuracy per-class
        acc_per_class = self.calculate_average_metric(
            logits, targets, lemmas, metric_name, average="none"
        )
        for i, acc_i in enumerate(acc_per_class):
            class_name = self.hparams.class_map[i]
            self.log(
                f"{prefix}{stage}_{metric_name}_{class_name}",
                acc_i,
                batch_size=batch_size,
            )

    @torch.no_grad()
    def calculate_average_metric(
        self,
        batch_logits: Tensor,
        batch_targets: List[List[LongTensor]],
        batch_lemmas: List[List[str]],
        metric_name: str,
        average: str = "micro",
    ) -> Tensor:
        """
        Calculates the mean accuracy based on the averaging method specified.

        batch_logits is B X MS X C
        batch_targets is B X S
        """
        # Extract sequence length from number of targets to predict
        sequence_lengths = [len(_) for _ in batch_targets]
        all_sense_ids = set(range(self.hparams.n_classes))

        metric_fn: Callable
        if metric_name == "acc":
            metric_fn = TF.accuracy
        elif metric_name == "f1":
            metric_fn = TF.f1_score

        for b, (logits, targets, s_len, lemmas) in enumerate(
            zip(batch_logits, batch_targets, sequence_lengths, batch_lemmas)
        ):
            # go through the lemmas in the sequence
            for l, lemma in enumerate(lemmas):
                possible_lemma_ids = self.hparams.lemma_to_sense_ids[lemma]
                impossible_lemma_ids = all_sense_ids - set(possible_lemma_ids)
                # ignore the impossible ids, by making their logits very low
                batch_logits[b, l, list(impossible_lemma_ids)] = -torch.inf

        # convert logits to preds, shape is B X MS
        preds = torch.argmax(batch_logits, dim=-1)

        # Calculate metric as the mean over all sentences
        metric = torch.vstack(
            [
                metric_fn(
                    preds=preds[batch_el, : sequence_lengths[batch_el]],
                    target=batch_targets[batch_el],
                    average=average,
                    num_classes=self.hparams.n_classes,
                    ignore_index=self.hparams.ignore_id,
                )
                for batch_el in range(len(batch_targets))
            ]
        ).nanmean(dim=0)

        return metric
