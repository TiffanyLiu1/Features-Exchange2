#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
# C:\Users\Pinky\PycharmProjects\FedAvg

# lr = 0.001, local_bs = 32


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from itertools import repeat
from collections import OrderedDict
import torch

from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from options import args_parser
from Update import LocalUpdate, DatasetSplit
from Nets import MLP, MLP_First, MLP_Second, MLP3Layers, CNNMnist, CNNCifar
from Fed import FedAvg
from test import test_img
from features import personal_avg, FtExch
from model_split import FirstModel, SecondModel, get_partial_weights


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            args.major_num = 400
            args.minor_num = 15
            dict_users, major_label = mnist_noniid(dataset_train, args.num_users, args.major_num, args.minor_num)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            args.major_num = 300
            args.minor_num = 20
            dict_users, major_label = cifar_noniid(dataset_train, args.num_users, args.major_num, args.minor_num)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_model = CNNCifar(args=args).to(args.device)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> add first & second model <<<<<<<<<<<<<<<<<<<<<<<<<<<<< NO NEEDED
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_model = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_model = MLP(dim_in=len_in, dim_hidden=args.hidden_units, dim_out=args.num_classes).to(args.device)
        #first_model = MLP_First(dim_in=len_in, dim_hidden=200).to(args.device)   .......................delete MLP_First & MLP_Second function
        #second_model = MLP_Second(dim_hidden=200, dim_out=args.num_classes).to(args.device)
    elif args.model == 'mlp3layers':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_model = MLP3Layers(dim_in=len_in, dim_hidden1=args.hidden1_units, dim_hidden2=args.hidden2_units, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_model)
    net_model.train()

    # copy weights
    w_glob = net_model.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    if not args.all_clients:
        print("NOT Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        avg_features = list(repeat(0, args.num_users))
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # get averaged features of all clients
        for idx in idxs_users:
            net_model.load_state_dict(w_locals[idx])
            first_model = FirstModel(args=args, net=net_model).to(args.device)
            images_num = args.major_num + (9 * args.minor_num)
            indiv_train = DataLoader(DatasetSplit(dataset_train, dict_users[idx]), batch_size=images_num, shuffle=True)
            for batch_idx, (images, labels) in enumerate(indiv_train):
                images, labels = images.to(args.device), labels.to(args.device)
                features = first_model(images)
            features_average = personal_avg(features, labels, major_label[idx], args)
            avg_features[idx] = features_average.float()

        # implement ftExch by using average features
        for idx in idxs_users:
            net_model.load_state_dict(w_locals[idx])
            second_model = SecondModel(args=args, net=net_model).to(args.device)
            local = FtExch(args=args, dataset=dataset_train, idxs=dict_users[idx], features=avg_features, userid=idx, chosen_users=idxs_users)
            w, loss = local.train(net=copy.deepcopy(net_model).to(args.device), sec=copy.deepcopy(second_model).to(args.device))
            w_locals[idx] = copy.deepcopy(w)
            loss_locals.append(copy.deepcopy(loss))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # update global weights
    if args.all_clients:
        w_glob = FedAvg(w_locals)
    else:
        # .................. EDIT ...........................
        # w_locals should append all the users that are considered in the ftExch, instead of just considering
        # the users in the last round
        w_locals_specific_users = []
        for idx in idxs_users:
            w_locals_specific_users.append(copy.deepcopy(w_locals[idx]))
        w_glob = FedAvg(w_locals_specific_users)
        # .................. EDIT ...........................

    # copy weight to net_glob
    net_model.load_state_dict(w_glob)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('C:/Users/Pinky/PycharmProjects/FedAvg/Plots/ftExch_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_model.eval()
    acc_train, loss_train = test_img(net_model, dataset_train, args)
    acc_test, loss_test = test_img(net_model, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
