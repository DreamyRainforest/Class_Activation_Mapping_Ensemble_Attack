""" Adapted VGG pytorch model that used as surrogate. """
import torchvision.models as models
import torch

import pdb
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary

class Resnet34(torch.nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.model = models.resnet34(pretrained=True).cuda().eval()
        # self.model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 10))
        self.features = list(self.model.children())[:-2]


    def prediction(self, x, internal=[]):
        with torch.no_grad():
            pred = self.model(x)
        if len(internal) == 0:
            return pred

        layers = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if (ii in internal):
                layers.append(x)
        return layers, pred

class Resnet18(torch.nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(pretrained=True).cuda().eval()
        # self.model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 10))
        self.features = list(self.model.children())[:-2]


    def prediction(self, x, internal=[]):
        with torch.no_grad():
            pred = self.model(x)
        if len(internal) == 0:
            return pred
        layers = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if (ii in internal):
                layers.append(x)
        return layers, pred


class Resnet152(torch.nn.Module):
    def __init__(self):
        super(Resnet152, self).__init__()
        self.model = models.resnet152(pretrained=True).cuda().eval()
        # self.model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 10))
        self.features = list(self.model.children())[:-2]

    def prediction(self, x, internal=[]):
        with torch.no_grad():
            pred = self.model(x)
        if len(internal) == 0:
            return pred

        layers = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if (ii in internal):
                layers.append(x)
        return layers, pred
