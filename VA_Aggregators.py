import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention
from Temporal_Attention import Temporal_Attention
from config import Config

class VA_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, a2e, f2e, embed_dim, cuda="cpu", va=True):
        super(VA_Aggregator, self).__init__()
        self.config = Config()
        self.va = va
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.a2e = a2e
        self.f2e = f2e
        #self.enc_attr = enc_attr
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Temporal_Attention(self.embed_dim)

    def forward(self, nodes, history_va, history_af):
        if self.config.debug:
            print("VA_Agg.py ,", "va =", self.va)
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(nodes)):
            history = history_va[i]
            famous = history_af[i]
            num_histroy_attr = len(history)
            if self.va == True:

                e_va = self.a2e.weight[history]
                e_af = self.f2e.weight[famous]
                va_rep = self.v2e.weight[nodes[i]]
            else:
                # attr component
                e_va = self.v2e.weight[history]
                va_rep = self.a2e.weight[nodes[i]]

            att_w = self.att(e_va, va_rep, e_af, num_histroy_attr)
            att_history = torch.mm(e_va.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats

