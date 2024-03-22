import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from config import Config

class Merge_Encoder(nn.Module):

    def __init__(self, attr_merge, enc_va_history, enc_ua_history, cuda="cpu", ua=True):
        super(Merge_Encoder, self).__init__()

        self.config = Config()
        self.attr_merge = attr_merge
        self.enc_va_history = enc_va_history
        self.enc_ua_history = enc_ua_history
        self.device = cuda

    def forward(self, nodes):
        if self.config.debug:
            print("Merge_Encoders.py")
        return self.attr_merge(self.enc_va_history(nodes),
                   self.enc_ua_history(nodes)).to(self.device)
