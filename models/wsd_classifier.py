from .base_classifier import BaseClassifier
from torch import nn, Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from typing import Tuple, Dict, List, Union

import torch
import torchmetrics.functional as TF


class WSDClassifier(BaseClassifier):
    def get_classifier_head(self, n_hidden: int, n_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, encoded_input: BatchEncoding, salient_idxs: Tensor) -> Tuple[Tensor, Tensor]:
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
            # filtering out non-target pos and lemmas
            "pos": [pos_tag[idx] for idx, pos_tag in zip(idxs, pos)],
            "lemmas": [lemma[idx] for idx, lemma in zip(idxs, lemmas)],
        }
        # everything is now only in terms of instances, we have filtered out wf.

    def log_metrics(self, processed_batch: Dict, stage: str, prefix: str = ""):
        # log F1 score overall and per pos-tag (not per class)
        batch_logits = processed_batch["logits"]
        batch_targets = processed_batch["targets"]
        batch_pos = processed_batch["pos"]

        batch_size = len(batch_logits)
        f1_avg = self.calc_avg_f1(batch_logits, batch_targets, batch_pos)
        self.log(f"{stage}_f1", f1_avg, batch_size=batch_size)

        if stage != "test":
            # No need to log per-pos-tag accuracy for train and val
            return

        # calculate & log average f1 per-pos tag
        f1_per_pos = self.calc_avg_f1(batch_logits, batch_targets, batch_pos, average="none")
        for i, f1_i in enumerate(f1_per_pos):
            pos_name = self.hparams.pos_map[i]
            self.log(f"{prefix}{stage}_acc_{pos_name}", f1_i, batch_size=batch_size)

    @torch.no_grad()
    def calc_avg_f1(
        self,
        batch_logits: Tensor,
        batch_targets: List[LongTensor],
        batch_pos: List[LongTensor],
        average: str = "micro",
    ):
        n_pos_tags = len(self.hparams.pos_map)
        # Extract sequence length from number of senses to predict
        seq_lens = [len(_) for _ in batch_targets]

        # TODO: only consider senses from current lemma in evaluation/inference
        ## WARNING: untested code
        batch_logits_per_pos = [
            torch.vstack(
                [
                    logits[:s_len][pos == pos_id]
                    for logits, pos, s_len in zip(batch_logits, batch_pos, seq_lens)
                ]
            )
            for pos_id in range(n_pos_tags)
        ]

        batch_senses_per_pos = [
            torch.cat([senses[pos == pos_id] for senses, pos in zip(batch_targets, batch_pos)])
            for pos_id in range(n_pos_tags)
        ]

        f1 = torch.vstack(
            [
                TF.f1_score(
                    preds=batch_logits_per_pos[pos_id],
                    target=batch_senses_per_pos[pos_id],
                    average=average,
                    num_classes=batch_logits.shape[-1],
                    ignore_index=self.hparams.ignore_id,
                )
                for pos_id in range(n_pos_tags)
                # there might be an issue for missing POS in batch
            ]
        ).nanmean(dim=0)

        return f1
