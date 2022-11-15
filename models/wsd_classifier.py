import sys
from .base_classifier import BaseClassifier
from torch import nn, Tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from typing import Set, Tuple, Dict, List, Union

import torch
import torchmetrics.functional as TF


class WSDClassifier(BaseClassifier):
    def __init__(
        self,
        pos_map: List[str],
        lemma_to_sense_ids: Dict[str, List[int]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(pos_map, lemma_to_sense_ids)

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

    def log_metrics(self, processed_batch: Dict, stage: str, prefix: str = ""):
        # log F1 score overall and per pos-tag (not per class)
        batch_logits = processed_batch["logits"]
        batch_targets = processed_batch["targets"]
        batch_pos = processed_batch["pos"]
        batch_lemmas = processed_batch["lemmas"]

        batch_size = len(batch_logits)
        all_sense_ids = set(range(self.hparams.n_classes))

        # NOTE: this is a shortcut i thought of - compute f1 per pos. then to get
        # average f1 we can just average that? previously we did calc_avg_f1 twice, once
        # with and once without averaging. I've changed default average to 'none'
        f1_per_pos = self.calc_avg_f1(
            batch_logits, batch_targets, batch_pos, batch_lemmas, all_sense_ids
        )
        f1_avg = f1_per_pos.nanmean()  # this is part of the shortcut
        self.log(f"{stage}_f1", f1_avg, batch_size=batch_size)

        if stage != "test":
            # No need to log per-pos-tag accuracy for train and val
            return

        # log average f1 per-pos tag
        for i, f1_i in enumerate(f1_per_pos):
            pos_name = self.hparams.pos_map[i]
            self.log(f"{prefix}{stage}_acc_{pos_name}", f1_i, batch_size=batch_size)

    @torch.no_grad()
    def calc_avg_f1(
        self,
        batch_logits: Tensor,
        batch_targets: List[LongTensor],
        batch_pos: List[LongTensor],
        batch_lemmas: List[List[str]],
        all_sense_ids: Set[int],
        average: str = "none",
    ):
        """Warning: this function is untested"""
        n_pos_tags = len(self.hparams.pos_map)
        # Extract sequence length from number of senses to predict
        seq_lens = [len(_) for _ in batch_targets]

        batch_logits_per_pos = [
            torch.vstack(
                [
                    logits[:s_len][pos == pos_id]
                    for logits, pos, s_len in zip(batch_logits, batch_pos, seq_lens)
                ]
            )
            for pos_id in range(n_pos_tags)
        ]

        batch_logits_per_pos = []
        for pos_id in range(n_pos_tags):
            logits_per_pos = []
            # go through batch
            for logits, pos, s_len, lemmas in zip(
                batch_logits,
                batch_pos,
                seq_lens,
                batch_lemmas,
            ):
                seq_logits = logits[:s_len]
                # go through sequence
                for i, lemma in enumerate(lemmas):
                    # set difference, to get the ids of irrelevant senses for lemma
                    non_lemma_ids = list(
                        all_sense_ids - set(self.hparams.lemma_to_sense_ids[lemma])
                    )
                    # make them irrelevant by masking their logits out
                    seq_logits[i, non_lemma_ids] = -torch.inf
                # only keep logits for current pos
                cur_pos_logits = seq_logits[pos == pos_id]
                logits_per_pos.append(cur_pos_logits)
            batch_logits_per_pos.append(torch.vstack(logits_per_pos))

        batch_senses_per_pos = [
            torch.cat(
                [
                    senses[pos == pos_id]
                    for senses, pos in zip(
                        batch_targets,
                        batch_pos,
                    )
                ]
            )
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
                # account for unseen pos in batch
                if len(batch_senses_per_pos[pos_id]) > 0
                else torch.full(
                    (batch_logits.shape[-1],),
                    fill_value=float("nan"),
                    device=self.device,
                )
                for pos_id in range(n_pos_tags)
            ]
            # dim=1: aggregate over all senses for each pos tag
        ).nanmean(dim=1)
        return f1
