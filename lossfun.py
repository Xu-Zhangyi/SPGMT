import torch
from torch.nn import Module
from setting import SetParameter
import torch.nn.functional as F
config = SetParameter()


class SpaLossFun(Module):
    def __init__(self, train_batch, distance_type):
        super(SpaLossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        self.flag = True

        # if self.flag:
        #     if str(config.dataset) == "tdrive" or "beijing" or "porto":
        #         if str(config.distance_type) == "TP":
        #             self.extra_coe = 4
        #         elif str(config.distance_type) == "DITA":
        #             if str(config.dataset) == "beijing":
        #                 self.extra_coe = 8
        #             if str(config.dataset) == "porto":
        #                 self.extra_coe = 4
        #         elif str(config.distance_type) == "LCRS":
        #             self.extra_coe = 16
        #         elif str(config.distance_type) == "discret_frechet":
        #             self.extra_coe = 4

        # if self.flag:
        #     if str(config["dataset"]) == "tdrive" or "beijing":
        #         if str(config["distance_type"]) == "TP":
        #             extra_coe = 0.5
        #         elif str(config["distance_type"]) == "DITA":
        #             extra_coe = 1
        #         elif str(config["distance_type"]) == "LCRS":
        #             extra_coe = 32
        #         elif str(config["distance_type"]) == "discret_frechet":
        #             extra_coe = 1

    def forward(self, embedding_a, embedding_p, embedding_n, pos_dis, neg_dis, device):

        # pos_dis = (pos_dis*self.extra_coe)
        # neg_dis = (neg_dis*self.extra_coe)

        D_ap = torch.exp(-pos_dis).to(device)  # -700
        D_an = torch.exp(-neg_dis).to(device)

        v_ap = torch.exp(-(torch.norm(embedding_a-embedding_p, p=2, dim=-1)))
        v_an = torch.exp(-(torch.norm(embedding_a-embedding_n, p=2, dim=-1)))
        loss_entire_ap = (D_ap - v_ap) ** 2
        loss_entire_an = (D_an - v_an) ** 2
        loss = loss_entire_ap + loss_entire_an + (D_ap > D_an)*(F.relu(v_an - v_ap)) ** 2
        loss_mean = loss.mean(dim=-1)
        return loss_mean

