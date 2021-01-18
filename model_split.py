#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
import torch.nn as nn
from collections import OrderedDict


class FirstModel(nn.Module):
    def __init__(self, args, net):
        super(FirstModel, self).__init__()
        self.args = args
        self.features = nn.Sequential(
            # stop at the features exchange layer
            *list(net.children())[:-(args.execlayer_num - args.ftExch_execlayer)]
        )
    def forward(self, x):
        x = self.features(x)
        return x


class SecondModel(nn.Module):
    def __init__(self, args, net):
        super(SecondModel, self).__init__()
        self.args = args
        self.features = nn.Sequential(
            # stop at the features exchange layer
            *list(net.children())[-(args.execlayer_num - args.ftExch_execlayer):]
        )
    def forward(self, x):
        x = self.features(x)
        return x


def get_partial_weights(args, w_net):
    # type of w_net is OrderedDict()
    # this case gets partial weights of the second model
    # can directly load these weights to the model by: #_model.load_state_dict(w_#)
    w_partial = OrderedDict()
    tuple_list = list(w_net.items())
    for i in range(args.ftExch_layer * 2, args.layer_num * 2):
        w_partial[tuple_list[i][0]] = tuple_list[i][1]
    return w_partial


