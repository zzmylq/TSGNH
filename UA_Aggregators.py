import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

from tqdm import tqdm

from Attention import Attention
from Temporal_Attention import Temporal_Attention
from Temporal_Attention_short import Temporal_Attention_short
from config import Config

class UA_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, t2e, enc_attr, embed_dim, cuda="cpu", ua=True):
        super(UA_Aggregator, self).__init__()
        self.config = Config()
        self.ua = ua
        self.temporal = self.config.temporal
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.t2e = t2e
        self.enc_attr = enc_attr
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        if self.temporal:
            self.att = Temporal_Attention(self.embed_dim)
        else:
            self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_ua, history_r, history_uat):
        if self.config.debug:
            print("UA_Agg.py ,", "ua =", self.ua)
        embed_matrix = torch.empty(len(history_ua), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_ua)):

            if history_ua[i] == []:
                continue
            history = history_ua[i]
            num_histroy_attr = len(history)
            tmp_label = history_r[i]
            tmp_time = history_uat[i]

            if self.ua == True:

                e_ua = self.enc_attr(torch.LongTensor(torch.tensor(list(history)).cpu().numpy())).to(self.device)

                ua_rep = self.u2e.weight[nodes[i]]
            else:

                e_ua = self.u2e.weight[history]
                ua_rep = self.enc_attr.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]

            if len(tmp_time) >= self.t2e.num_embeddings:
                for k in range(len(tmp_time)):
                    if tmp_time[k] >= self.t2e.num_embeddings:
                        tmp_time[k] = self.t2e.num_embeddings - 1
            e_t = self.t2e.weight[tmp_time]
            x = torch.cat((e_ua, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))
            if self.temporal:
                att_w = self.att(o_history, ua_rep, e_t, num_histroy_attr)
            else:
                att_w = self.att(o_history, ua_rep, num_histroy_attr)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()
            embed_matrix[i] = att_history

        to_feats = embed_matrix
        return to_feats
