import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F
from config import Config

class Temporal_Attention_short(nn.Module):
    def __init__(self, embedding_dims):
        super(Temporal_Attention_short, self).__init__()
        self.config = Config()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)

        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.att4 = nn.Linear(self.embed_dim * 2, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, e_t, num_neighs):
        last_node = node1[0]
        last_reps = last_node.repeat(num_neighs, 1)

        if self.config.sample_model:
            x = torch.cat((node1, last_reps), 1)
            x = self.att4(x)
            att = F.softmax(x, dim=0)
        else:
            x = torch.cat((node1, last_reps), 1)
            x = F.relu(self.att1(x))
            x = F.dropout(x, training=self.training)
            x = F.relu(self.att2(x))
            x = F.dropout(x, training=self.training)
            x = self.att3(x)
            att = F.softmax(x, dim=0)
        return att

