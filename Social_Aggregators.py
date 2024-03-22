import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from Attention import Attention
from config import Config

class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, u2e, embed_dim, cuda="cpu"):
        super(Social_Aggregator, self).__init__()
        self.config = Config()
        self.device = cuda
        #self.enc_user = enc_user
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        if self.config.debug:
            print("Social_Agg.py")
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            #
            tmp_list_list = list(tmp_adj)
            e_u = self.u2e.weight[tmp_list_list]
            u_rep = self.u2e.weight[nodes[i]]

            att_w = self.att(e_u, u_rep, num_neighs)
            if self.config.print_social_attention:
                print("social_att:")
                tt_list = []
                aa = att_w.tolist()
                for aaa in range(len(aa)):
                    tt_list.append(aa[aaa][0])
                for p,q in zip(tmp_list_list, tt_list):
                    print(p,q)

            att_history = torch.mm(e_u.t(), att_w).t()
            embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats
