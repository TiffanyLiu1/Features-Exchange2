#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
from torch import nn
import torch.nn.functional as F


# 1 2 4 5
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten(1, -1)
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


# .....................................  optional ....................
class MLP_First(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super(MLP_First, self).__init__()
        self.flatten = nn.Flatten(1, -1)
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.flatten(x)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class MLP_Second(nn.Module):
    def __init__(self, dim_hidden, dim_out):
        super(MLP_Second, self).__init__()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.layer_hidden(x)
        return x
# .....................................  optional ....................


# 2 3 7 8
class MLP3Layers(nn.Module):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_out):
        super(MLP3Layers, self).__init__()
        self.flatten = nn.Flatten(1, -1)
        self.layer_input = nn.Linear(dim_in, dim_hidden1)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.layer_hidden1 = nn.Linear(dim_hidden1, dim_hidden2)
        self.dropout1 = nn.Dropout()
        self.relu1 = nn.ReLU()
        self.layer_hidden2 = nn.Linear(dim_hidden2, dim_out)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.layer_hidden2(x)
        return x


# 3 4 11 12
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2_drop(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 3 5 9 12
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


# # original function
# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, args.num_classes)
#         self.execlayer_num = 12
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x


# # original function
# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#         self.execlayer_num = 12
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
