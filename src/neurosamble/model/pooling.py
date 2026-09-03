import math
from typing import Dict, Literal, Optional, Tuple, Type

import torch
import torch.nn as nn

from dna2vec.tokenizer  import BPTokenizer

import logging

class AveragePooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    # derived from https://github.com/princeton-nlp/SimCSE/blob/13361d0e29da1691e313a94f003e2ed1cfa97fef/simcse/models.py#LL49C1-L84C1
    """

    def __init__(self):
        super().__init__()

    def forward(self, last_hidden, attention_mask):
        # Old previous implementation
        return (last_hidden * attention_mask.unsqueeze(-1)).sum(
            1
        ) / attention_mask.sum(-1).unsqueeze(-1)

        # last_hidden.names = ["batch", "sequence", "embedding"]
        # attention_mask.names = ["batch", "sequence"]
        # # using named tensors
        # attention_mask = attention_mask.align_to(
        #     "batch", "sequence", "embedding"
        # )  # eq. to unsqueeze

        # avg_emb = (last_hidden * attention_mask).sum("sequence") / attention_mask.sum(
        #     "sequence"
        # )
        # return avg_emb

