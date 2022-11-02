from typing import Tuple, Dict

from torch.nn.utils.rnn import pad_sequence

from .base_classifier import BaseClassifier
from torch import nn, Tensor
from transformers import BatchEncoding


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
        }

    def log_metrics(self, processed_batch: Dict, stage: str):
        # log F1 score overall and per pos-tag (not per class)
        pass
