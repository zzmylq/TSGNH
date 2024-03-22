import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from config import Config

class UV_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_uv_lists, history_r_lists, history_ut_lists, aggregator, cuda="cpu", uv=True):
        super(UV_Encoder, self).__init__()

        self.config = Config()
        self.features = features
        self.uv = uv
        self.history_uv_lists = history_uv_lists
        self.history_r_lists = history_r_lists
        self.history_ut_lists = history_ut_lists
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        if self.config.long_short == "both":
            self.linear1 = nn.Linear(3 * self.embed_dim, self.embed_dim)
        else:
            self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        if self.config.debug:
            print("UV_Encoders.py ,", "uv =", self.uv)
        tmp_history_uv = []
        tmp_history_r = []
        tmp_history_ut = []
        for node in nodes:
            tmp_history_uv.append(self.history_uv_lists[int(node)])
            tmp_history_r.append(self.history_r_lists[int(node)])
            tmp_history_ut.append(self.history_ut_lists[int(node)])

        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_r, tmp_history_ut)
        self_feats = self.features.weight[nodes]

        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
