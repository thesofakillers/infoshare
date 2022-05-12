from argparse import ArgumentParser
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, BatchEncoding
from typing import List, Tuple

import numpy as np
import torch


class BERTEncoderForWordClassification(Module):
    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("BERT Encoder")
        parser.add_argument(
            "--encoder_name",
            type=str,
            default="roberta-base",
            help="The name of the HuggingFace model to use as an encoder.",
        )
        parser.add_argument(
            "--aggregation",
            type=str,
            default="mean",
            choices=["first", "max", "mean"],
            help="The method for wordpiece embedding aggregation.",
        )
        parser.add_argument(
            "--probe_layer",
            type=int,
            default=-1,
            help="The index of the encoder layer to output for probing.",
        )
        return parent_parser

    def __init__(
        self,
        encoder_name: str,
        aggregation: str = "mean",
        probe_layer: int = -1,
        **kwargs,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.aggregation = aggregation
        self.probe_layer = probe_layer

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
        ).hidden_states[self.probe_layer]

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

            # Append sequence output to batch outputgit
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

    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @staticmethod
    def _get_tokens_per_word(word_ids: List[int]) -> Tuple[List[int], Tuple[int, int]]:
        """Takes a list with the word ids for each (sub-word) token in a sequence and 
        returns the number of tokens each word has as well as the start and end token
        of the sequence without special / padding tokens.
        
        Arguments:
            word_ids(List[int]): The word ids for each (sub-word) token in a sequence.
            
        Returns:
            Tuple[List[int], Tuple[int, int]]: The number of tokens each word has 
            (first part of the tuple) as well as the actual start and end token of the
            sequence (second part of the tuple).
            
        For example, if the sequence length is 10 and the word_ids are:
        [None, 0, 0, 0, 1, 2, 2, None, None, None]
        Then the tuple returned is:
        ([3, 1, 2], (1, 7))
        because the first word (with id 0) has 3 tokens, the second word (with id 1)
        has 1 token and the third word (with id 2) has 2 tokens. The actual sequence
        start at the 1st token and finishes before the 7th token, since None word ids
        correspond to special or padding tokens.
        """
        diff = lambda x: np.diff(np.concatenate([[0], x, [np.max(x) + 1]]))

        # Get sequence boundaries (excluding special tokens)
        seq = [1 if word_id is not None else 0 for word_id in word_ids]
        seq_start, seq_end, _ = np.where(diff(seq))[0]

        # Calculate the amount of sub-words corresponding to each word
        word_ids = np.array(word_ids[seq_start:seq_end]) + 1
        seq_lengths = diff(np.where(diff(word_ids))[0])[1:-1]

        return seq_lengths.tolist(), (seq_start, seq_end)
