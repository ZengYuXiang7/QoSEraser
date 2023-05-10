import numpy as np
from numpy import *
import pandas as pd
import torch as t

from datasets.balance_paritition import balance_part
from models.clusters import k_mean, deep_cluster
from models.graph_embedding import get_user_embedding, get_item_embedding
from utility.utils import *
# from datasets.dataset import FeatureLookup
import pickle as pk
import dgl as d
from tqdm import *


# node2vec
def eraser(datasets, args):

    log('\t图嵌入非均衡切割')
    try:
        # label = pk.load(open(f'./pretrain/non_balance_user_based_{args.slices}.pk', 'rb'))
        label = pk.load(open(f'./pretrain/non_132213balance_user_based_{args.slices}.pk', 'rb'))
    except IOError:

        user_embedding = get_user_embedding(args)
        if args.part_type == 3:
            label = k_mean(user_embedding, args.slices, args)
        elif args.part_type == 5:
            label = deep_cluster(user_embedding, args)

        pk.dump(label, open(f'./pretrain/non_balance_user_based_{args.slices}.pk', 'wb'))

    label = np.array(label)
    # print(label)

    # 检查每个类的个数
    dic = {}
    for i in label:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] = dic[i] + 1
    print(dic)

    # 获得切分矩阵
    split_train_Tensor = []
    for i in range(args.slices):
        split_tensor = np.zeros_like(datasets)
        Idx = (label == i).nonzero()[0]
        for idx in Idx:
            split_tensor[idx] = datasets[idx]
        split_train_Tensor.append(split_tensor)

    return split_train_Tensor, label


# 均衡切割
def eraser4_2(datasets, args):
    # log('\t图嵌入均衡切割')

    try:
        C = pk.load(open(f'./pretrain/user_based_{args.slices}.pk', 'rb'))
    except:
        user_embedding = get_user_embedding(args)

        label_1 = None
        if args.cluster == 'kmean':
            label_1 = k_mean(user_embedding, args.slices, args)
        elif args.cluster == 'deep':
            label_1 = deep_cluster(user_embedding, args)

        C = balance_part(label_1, user_embedding, args)
        pk.dump(C, open(f'./pretrain/user_based_{args.slices}.pk', 'wb'))

    label = [0 for _ in range(339)]
    for i in range(len(C)):
        for j in C[i]:
            label[j] = i
        print(len(C[i]), end = ' ')
    print()

    train_tensor, valid_tensor, test_tensor = datasets
    split_train_tensor = []
    for i in range(args.slices):
        # idx =
        idx = np.array(C[i], dtype='int32')
        temp = np.zeros_like(train_tensor)
        temp[idx] = train_tensor[idx]
        split_train_tensor.append(temp)

    split_valid_tensor = []
    for i in range(args.slices):
        # idx =
        idx = np.array(C[i], dtype='int32')
        temp = np.zeros_like(test_tensor)
        temp[idx] = valid_tensor[idx]
        split_valid_tensor.append(temp)

    split_test_tensor = []
    for i in range(args.slices):
        # idx =
        idx = np.array(C[i], dtype='int32')
        temp = np.zeros_like(test_tensor)
        temp[idx] = test_tensor[idx]
        split_test_tensor.append(temp)

    return split_train_tensor, split_valid_tensor, split_test_tensor, label


# node2vec 基于服务切分 非平衡
def eraser5(datasets, args):

    try:
        C = pk.load(open(f'./pretrain/item_based_{args.slices}.pk', 'rb'))
    except IOError:
        item_embedding = get_item_embedding(args)
        print(item_embedding.shape)
        label_1 = None
        if args.cluster == 'kmean':
            label_1 = k_mean(item_embedding, args.slices, args)
        elif args.cluster == 'deep':
            label_1 = deep_cluster(item_embedding, args)

        C = balance_part(label_1, item_embedding, args)
        pk.dump(C, open(f'./pretrain/item_based_{args.slices}.pk', 'wb'))

    label = [0 for _ in range(5825)]
    for i in range(len(C)):
        for j in C[i]:
            label[j] = i
        print(len(C[i]), end=' ')
    print()

    train_tensor, valid_tensor, test_tensor = datasets
    split_train_tensor = []
    for i in range(args.slices):
        # idx =
        idx = np.array(C[i], dtype='int32')
        temp = np.zeros_like(train_tensor)
        temp[:, idx] = train_tensor[:, idx]
        split_train_tensor.append(temp)

    split_valid_tensor = []
    for i in range(args.slices):
        # idx =
        idx = np.array(C[i], dtype='int32')
        temp = np.zeros_like(test_tensor)
        temp[:, idx] = valid_tensor[:, idx]
        split_valid_tensor.append(temp)

    split_test_tensor = []
    for i in range(args.slices):
        # idx =
        idx = np.array(C[i], dtype='int32')
        temp = np.zeros_like(test_tensor)
        temp[:, idx] = test_tensor[:, idx]
        split_test_tensor.append(temp)

    return split_train_tensor, split_valid_tensor, split_test_tensor, label


# 地区切割， 效果巨差！！！！！！！！！！！！！
def eraser3(datasets, args):
    log('\t简单聚类')
    df = pd.read_csv('./datasets/data/WSDREAM/userlist_idx.csv').to_numpy()
    x = []
    for i in range(df.shape[0]):
        x.append([df[i][3], df[i][5]])
    x = np.array(x)

    label = k_mean(x, args.slices, args)

    # 检查每个类的个数
    dic = {}
    for i in label:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] = dic[i] + 1
    print(dic)

    C = label
    train_tensor, valid_tensor, test_tensor = datasets
    split_train_tensor = []
    for i in range(args.slices):
        # idx =
        idx = np.array(C[i], dtype='int32')
        temp = np.zeros_like(train_tensor)
        temp[idx] = train_tensor[idx]
        split_train_tensor.append(temp)

    split_valid_tensor = []
    for i in range(args.slices):
        # idx =
        idx = np.array(C[i], dtype='int32')
        temp = np.zeros_like(test_tensor)
        temp[idx] = valid_tensor[idx]
        split_valid_tensor.append(temp)

    split_test_tensor = []
    for i in range(args.slices):
        # idx =
        idx = np.array(C[i], dtype='int32')
        temp = np.zeros_like(test_tensor)
        temp[idx] = test_tensor[idx]
        split_test_tensor.append(temp)

    return split_train_tensor, split_valid_tensor, split_test_tensor, label


# node2vec 基于服务切分
def eraser2(datasets, args):
    label = None
    try:
        label = pk.load(open(f'./pretrain/item_based_{args.slices}.pk', 'rb'))
    except IOError:
        item_embedding = get_item_embedding(args)

        label = k_mean(item_embedding, args.slices, args)

        pk.dump(label, open(f'./pretrain/item_based_{args.slices}.pk', 'wb'))

    split_train_Tensor = []

    # 获得切分矩阵
    for i in range(args.slices):
        split_tensor = np.zeros_like(datasets)
        Idx = (label == i).nonzero()[0]
        for idx in Idx:
            split_tensor[:, idx] = datasets[:, idx]
        split_train_Tensor.append(split_tensor)

    return split_train_Tensor, label


