from collections import defaultdict
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, BatchEncoding
from typing import List

import torch
import torch.nn as nn

class BERTForWordClassification(nn.Module):
    def __init__(self, encoder_name: str):
        super().__init__()
        
        # load the pretrained model
        self.model = AutoModel.from_pretrained(encoder_name)

        # freeze BERT's parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, encoded_input: BatchEncoding) -> Tensor:
        batch_output = []

        # TODO: change last hidden state
        bert_output = self.model(**encoded_input).last_hidden_state
        
        # iterate over batch samples
        for sample_idx, sample_outputs in enumerate(bert_output):
            # a list indicating the word corresponding to each token
            word_ids = encoded_input.word_ids(sample_idx)
            
            # group token representations by word
            word_representations = defaultdict(list)
            for tensor, token_id in zip(sample_outputs, word_ids):
                word_representations[token_id] += [tensor]

            # remove redundant tensors for special tokens
            del word_representations[None]

            # TODO: add other options
            # aggregate embeddings per word
            avg_tensors = {
                t_id: torch.vstack(w_t).mean(dim=0)
                for t_id, w_t in word_representations.items()
            }
            
            # order tensors by their corresponding word id & pad with zeros
            max_word_id = max([word_id for word_id in word_ids if word_id])
            ordered_tensors = torch.vstack(
                [avg_tensors[i] for i in range(max_word_id + 1)]
            )

            batch_output += [ordered_tensors]

        # pad sequence representations
        batch_output = pad_sequence(batch_output, batch_first=True)
        
        return batch_output
