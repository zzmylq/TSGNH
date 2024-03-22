import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.config = Config()
    def forward(self, scores, labels_list, atanh_alpha, x_u, x_v, device):
        mf_loss = labels_list * torch.log(scores + 1e-9) + (1 - labels_list) * torch.log(1 - scores - 1e-9)

        ones = torch.ones([self.config.hash_code_length, 1]).to(device)

        I_n = torch.eye(x_u.shape[0]).to(device)
        I_m = torch.eye(x_v.shape[0]).to(device)

        loss_B1 = torch.norm(torch.matmul(x_u, ones))**2
        loss_D1 = torch.norm(torch.matmul(x_v, ones))**2
        loss_BB = torch.norm(torch.matmul(x_u, x_u.T) - self.config.hash_code_length * I_n)**2
        loss_DD = torch.norm(torch.matmul(x_v, x_v.T) - self.config.hash_code_length * I_m)**2

        if self.config.balance and self.config.decorrelation:
            loss = -torch.sum(mf_loss) + self.config.atanh_alpha_weight * (1 / atanh_alpha) \
                   + self.config.balance_weight * \
                   (loss_B1 + loss_D1) \
                   + self.config.decorrelation_weight * \
                    (loss_BB + loss_DD)
        if self.config.balance and not self.config.decorrelation:
            loss = -torch.sum(mf_loss) + self.config.atanh_alpha_weight * (1 / atanh_alpha) \
                   + self.config.balance_weight * \
                     (loss_B1 + loss_D1)
        if not self.config.balance and self.config.decorrelation:
            loss = -torch.sum(mf_loss) + self.config.atanh_alpha_weight * (1 / atanh_alpha) \
                   + self.config.decorrelation_weight * \
                     (loss_BB + loss_DD)
        if not self.config.balance and not self.config.decorrelation:
            loss = -torch.sum(mf_loss) + self.config.atanh_alpha_weight * (1 / atanh_alpha)

        return loss