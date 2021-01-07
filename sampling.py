#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8


import numpy as np
from torchvision import datasets, transforms

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users

# # delete >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# for num in range(10):
#     count = 0
#     for i in range(len(labels)):
#         if labels[i] == num:
#             count += 1
#     print('count: ', count)
#
# train_filter = np.where(labels == 0)
# print('train_filter: ', train_filter)
# print('length of train_filter: ', len(train_filter[0]))
# # delete >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# # edit >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE
def mnist_noniid(dataset, num_users, major_num, minor_num):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    labels = dataset.train_labels.numpy()

    # divide and assign
    major_label = list(range(0, 10)) * 10
    for lb in range(10):
        label_filter = np.where(labels == lb)
        label_filter = np.asarray(label_filter[0])
        for i in range(num_users):
            if major_label[i] != lb:
                amount = minor_num
            else:
                amount = major_num
            dict_users[i] = np.concatenate((dict_users[i], label_filter[0:amount]), axis=0)
            label_filter = list(set(label_filter) - set(label_filter[0:amount]))

    # # check ...........................delete
    # print(dict_users[1])
    # print(labels[16385])
    # for n in range(10):
    #     count = 0
    #     for c in dict_users[93]:
    #         if labels[c] == n:
    #             count = count + 1
    #     print(count)
    # for i in dict_users[0]:
    #     b = list(range(1, 100))
    #     for n in b:
    #         for j in dict_users[n]:
    #             if i == j:
    #                 print('False')
    # ....................................delete
    return dict_users, major_label
# # edit >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)