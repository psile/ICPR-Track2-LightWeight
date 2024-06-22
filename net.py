from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

import os

# from model import ACM
from model.net import *
class LightWeightNetwork(nn.Module):
    def __init__(self,):
        super(LightWeightNetwork, self).__init__()
       
        #pdb.set_trace()
        
        self.model = LightWeightNetwork1()
       
        
    def forward(self, img):
        return self.model(img)

    # def loss(self, pred, gt_mask):
    #     loss = self.cal_loss(pred, gt_mask)
    #     return loss
