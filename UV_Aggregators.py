import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention
from Temporal_Attention import Temporal_Attention
from Temporal_Attention_short import Temporal_Attention_short
from config import Config

class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, t2e, embed_dim, cuda="cpu", uv=True):
        super(UV_Aggregator, self).__init__()
        self.config = Config()
        self.uv = uv
        self.temporal = self.config.temporal
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.t2e = t2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        if self.temporal:
            self.att = Temporal_Attention(self.embed_dim)
            self.att_short = Temporal_Attention_short(self.embed_dim)
        else:
            self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_uv, history_r, history_uvt):
        if self.config.debug:
            print("UV_Agg.py ,", "uv =", self.uv)
        if self.config.long_short == "both":
            embed_matrix = torch.empty(len(history_uv), self.embed_dim * 2, dtype=torch.float).to(self.device)
        else:
            embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_uv)):
            if history_uv[i] == []:
                continue
            history = history_uv[i]
            num_histroy_item = len(history)
            tmp_label = history_r[i]
            tmp_time = history_uvt[i]

            if self.uv == True:
                # user component
                e_uv = self.v2e.weight[history]
                uv_rep = self.u2e.weight[nodes[i]]
            else:
                # item component
                e_uv = self.u2e.weight[history]
                uv_rep = self.v2e.weight[nodes[i]]

            e_r = self.r2e.weight[tmp_label]
            e_t = self.t2e.weight[tmp_time]
            x = torch.cat((e_uv, e_r), 1)
            if self.config.sample_model:
                o_history = F.relu(self.w_r1(x))
            else:
                x = F.relu(self.w_r1(x))
                o_history = F.relu(self.w_r2(x))
            if self.config.long_short == "both" or self.config.long_short == "long":

                if self.temporal:
                    att_w = self.att(o_history, uv_rep, e_t, num_histroy_item)
                    if self.config.print_interest_attention:
                        print("att_w",att_w)
                else:
                    att_w = self.att(o_history, uv_rep, num_histroy_item)
                att_history = torch.mm(o_history.t(), att_w)
                att_history = att_history.t()

            if self.config.long_short == "both" or self.config.long_short == "short":

                if self.temporal:
                    att_ws = self.att_short(o_history, uv_rep, e_t, num_histroy_item)
                    if self.config.print_interest_attention:
                        print("att_ws",att_ws)
                    att_history_s = torch.mm(o_history.t(), att_ws)
                    att_history_s = att_history_s.t()
                else:
                    att_history_s = o_history[0]
                    att_history_s = att_history_s.reshape([1, self.embed_dim])

            if self.config.long_short == "both":
                embed_matrix[i] = torch.cat([att_history, att_history_s], dim=1)
            elif self.config.long_short == "long":
                embed_matrix[i] = att_history
            else:
                embed_matrix[i] = att_history_s
        to_feats = embed_matrix
        return to_feats
