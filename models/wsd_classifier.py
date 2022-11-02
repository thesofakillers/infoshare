from typing import Tuple, Dict

from torch.nn.utils.rnn import pad_sequence

from .base_classifier import BaseClassifier
from torch import nn, Tensor
import torch
from transformers import BatchEncoding
import torchmetrics.functional as TF


class WSDClassifier(BaseClassifier):
    def get_classifier_head(self, n_hidden: int, n_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, encoded_input: BatchEncoding, salient_idxs) -> Tuple[Tensor, Tensor]:
        embeddings = self.bert(encoded_input)
        # need to index the (padded) embeddings with salient idxs
        embeddings = embeddings.gather(1, salient_idxs)
        # indexing breaks padding, so pad again
        embeddings = pad_sequence(embeddings, batch_first=True)
        logits = self.classifier(embeddings)
        return embeddings, logits

    def process_batch(self, batch: Tuple) -> Dict[str, Tensor]:
        encodings, targets, idxs, pos, lemmas = batch
        embeddings, logits = self(encodings, idxs)  # (bs, max_seq_len, -1)
        return {
            "embeddings": embeddings,
            "logits": logits,
            "targets": targets,
            "pos": pos,
            "lemmas": lemmas,
        }

    def log_metrics(self, processed_batch: Dict, stage: str, prefix: str = ""):
        # log F1 score overall and per pos-tag (not per class)
        batch_logits = processed_batch["logits"]
        batch_targets = processed_batch["targets"]
        self.log_f1(batch_logits, batch_targets, stage=stage, prefix=prefix)

    def log_f1(self, logits: Tensor, targets: Tensor, stage: str, prefix: str = ""):
        batch_size = len(logits)

        f1_avg = self.calc_ave_f1(logits, targets)
        self.log(f"{stage}_f1", f1_avg, batch_size=batch_size)

        if stage != "test":
            # No need to log per-pos-tag accuracy for train and val
            return

        # calculate & log average accuracy per-pos tag
        acc_per_pos = self.calcu_ave_f1(logits, targets, average="none")
        for i, acc_i in enumerate(acc_per_pos):
            pos_name = self.hparams.pos_map[i]
            self.log(f"{prefix}{stage}_acc_{pos_name}", acc_i, batch_size=batch_size)

    @torch.no_grad()
    def calc_ave_f1(logits, targets, average: str = "micro"):
        # Extract sequence length from number of senses to predict
        sequence_lengths = [len(_) for _ in targets]

        # TODO: figure out how to do this PER POS-TAG.
        # TODO: only consider senses from current lemma in evaluation/inference
        # I think maybe an extra or different loop is necessary
        # In fact we might have to fully re-write this manually
        # calc f1 as mean over sentences
        f1 = torch.vstack(
            [
                TF.f1(
                    preds=logits[i, : sequence_lengths[i], :],
                    target=targets[i],
                    average=average,
                    num_classes=logits.shape[-1],
                    ignore_index=self.hparams.ignore_id,
                )
                for i in range(len(targets))
            ]
        ).nanmean(dim=0)

        return f1
