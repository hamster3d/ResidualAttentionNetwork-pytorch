from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
import time
# from model.residual_attention_network_pre import ResidualAttentionModel
# based https://github.com/liudaizong/Residual-Attention-Network
from model.residual_attention_network import ResidualAttentionModel_92_32input_update as ResidualAttentionModel

if __name__ == "__main__":

    model_file = 'model_92_sgd.pkl'

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = ResidualAttentionModel()
    print(model)

    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()