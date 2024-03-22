import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from config import Config

class VA_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_va_lists, history_af_lists, aggregator, cuda="cpu", va=True):
        super(VA_Encoder, self).__init__()
        self.config = Config()
        self.features = features
        self.va = va
        self.history_va_lists = history_va_lists
        self.history_af_lists = history_af_lists
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        if self.config.debug:
            print("VA_Encoders.py ,", "va =", self.va)
        tmp_history_va = []
        tmp_history_af = []
        for node in nodes:
            tmp_history_va.append(self.history_va_lists[int(node)])

        for attrs in tmp_history_va:
            this_attr_famous = []
            for attr in attrs:
                this_attr_famous.append(self.history_af_lists[attr])
            tmp_history_af.append(this_attr_famous)

        neigh_feats = self.aggregator.forward(nodes, tmp_history_va, tmp_history_af)
        self_feats = self.features.weight[nodes]

        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
