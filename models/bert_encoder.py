from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, BatchEncoding
from typing import List, Tuple

import numpy as np
import torch


class BERTEncoderForWordClassification(Module):
    def __init__(self, encoder_name: str, aggregation: str = "mean", output_layer: int = -1):
        super().__init__()

        self.aggregation = aggregation
        self.output_layer = output_layer

        # Load the pretrained model
        self.model = AutoModel.from_pretrained(encoder_name)

        # Freeze BERT's parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, batch_encoding: BatchEncoding) -> Tensor:
        # Get BERT representations for each (sub-word) token
        batch_output_per_token = self.model(
            **batch_encoding,
            output_hidden_states=True,
        ).hidden_states[self.output_layer]

        # Aggregate the BERT representations for each word
        batch_output_per_word = []
        for sequence_idx, sequence_output_per_token in enumerate(batch_output_per_token):
            # Get amount of (sub-word) tokens corresponding to each word
            word_ids = batch_encoding.word_ids(sequence_idx)
            split_idx, (start, end) = self._get_tokens_per_word(word_ids)

            # Split, aggregate and concatenate BERT representations
            sequence_output_per_word = torch.split(sequence_output_per_token[start:end], split_idx)
            sequence_output_per_word = list(map(self.aggregation_fn, sequence_output_per_word))
            sequence_output_per_word = torch.vstack(sequence_output_per_word)

            # Append sequence output to batch output
            batch_output_per_word += [sequence_output_per_word]

        # Pad sequence representations
        batch_output_per_word = pad_sequence(batch_output_per_word, batch_first=True)

        return batch_output_per_word

    def aggregation_fn(self, x: Tensor) -> Tensor:
        if self.aggregation == "mean":
            return torch.mean(x, dim=0)
        elif self.aggregation == "max":
            return torch.max(x, dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregation}")

    @staticmethod
    def _get_tokens_per_word(word_ids: List[int]) -> Tuple[List[int], Tuple[int, int]]:
        # TODO @tolis: documentation for `diff`
        diff = lambda x: np.diff(np.concatenate([[0], x, [np.max(x) + 1]]))

        # Get sequence boundaries (excluding special tokens)
        seq = [1 if word_id else 0 for word_id in word_ids]
        seq_start, seq_end, _ = np.where(diff(seq))[0]

        # Calculate the amount of sub-words corresponding to each word
        word_ids = np.array(word_ids[seq_start:seq_end]) + 1
        seq_lengths = diff(np.where(diff(word_ids))[0])[1:-1]

        return seq_lengths.tolist(), (seq_start, seq_end)
