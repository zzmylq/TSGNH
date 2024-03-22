import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from config import Config

class UA_Encoder(nn.Module):

    def __init__(self, features, embed_dim, history_ua_seq, history_uar_seq, history_uat_seq, aggregator, cuda="cpu", ua=True):
        super(UA_Encoder, self).__init__()

        self.config = Config()
        self.features = features
        self.ua = ua
        self.history_ua_seq = history_ua_seq
        self.history_uar_seq = history_uar_seq
        self.history_uat_seq = history_uat_seq
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        if self.config.debug:
            print("UA_Encoders.py ,", "ua =", self.ua)
        tmp_history_ua = []
        tmp_history_r = []
        tmp_history_ut = []
        for node in nodes:
            tmp_history_ua.append(self.history_ua_seq[int(node)])
            tmp_history_r.append(self.history_uar_seq[int(node)])
            tmp_history_ut.append(self.history_uat_seq[int(node)])
        neigh_feats = self.aggregator.forward(nodes, tmp_history_ua, tmp_history_r, tmp_history_ut)

        self_feats = self.features.weight[nodes]
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
