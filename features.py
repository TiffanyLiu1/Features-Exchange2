#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import torch
import torch.nn as nn
import numpy as np
import copy
from torch.utils.data import DataLoader
from torch.autograd import Variable
from itertools import repeat
from torchvision import models

from Update import DatasetSplit



def personal_avg(features, labels, major_label, args):
    ft_avg = torch.zeros([10, args.hidden_units], dtype=torch.float64)
    for num in range(10):
        label_filter = np.where(labels == num)
        label_filter = np.asarray(label_filter[0])
        for idx in label_filter:
            ft_avg[num] = ft_avg[num] + features[idx]
        if num != major_label:
            ft_avg[num] = ft_avg[num] / args.minor_num
        else:
            ft_avg[num] = ft_avg[num] / args.major_num
    return ft_avg


class FtExch(object):
    def __init__(self, args, dataset=None, idxs=None, features=None, userid=None, chosen_users=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.features = features
        self.userid = userid
        self.chosen_users = chosen_users
        self.target = torch.tensor(list(range(10)))

    def train(self, net, sec):
        # use averaged features to get other-features gradients
        grads_combo = []
        for i in self.chosen_users:
            if i != self.userid:
                log_probs = sec(Variable(self.features[i]))
                loss = self.loss_func(log_probs, self.target)
                loss.backward()
                grads_each = []
                params = list(sec.parameters())
                for indx in range((self.args.layer_num - self.args.ftExch_layer) * 2):
                    grads_each.append(params[indx].grad)
                grads_combo.append(copy.deepcopy(grads_each))

        # train locally and update with other-features gradients sum
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                # sum up local gradients with other-features gradients to get gradients sum
                params = list(net.parameters())
                for net_paramidx in range(self.args.ftExch_layer * 2, self.args.layer_num * 2):
                    sec_paramidx = net_paramidx - (self.args.ftExch_layer * 2)
                    for i in range(len(grads_combo)):
                        params[net_paramidx].grad = params[net_paramidx].grad + grads_combo[i][sec_paramidx]

                # update net model by gradients sum
                optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
