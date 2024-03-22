import torch
import torch.nn as nn
import torch.nn.functional as F


class Merge(nn.Module):
    def __init__(self, embedding_dims):
        super(Merge, self).__init__()
        self.embed_dim = embedding_dims
        self.layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.layer2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.softmax = nn.Softmax(0)

    def forward(self, y, z):
        x = torch.cat((y, z), 1)
        x = F.relu(self.layer1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.layer2(x))
        x = F.dropout(x, training=self.training)
        return x
