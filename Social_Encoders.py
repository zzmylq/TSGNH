import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from config import Config

class Social_Encoder(nn.Module):

    def __init__(self, features, embed_dim, social_adj_lists, aggregator, cuda="cpu"):
        super(Social_Encoder, self).__init__()
        self.config = Config()
        self.features = features
        self.social_adj_lists = social_adj_lists
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes):
        if self.config.debug:
            print("Social_Encoder.py!")

        to_neighs = []
        for node in nodes:
            to_neighs.append(self.social_adj_lists[int(node)])
        neigh_feats = self.aggregator.forward(nodes, to_neighs)

        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)

        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
