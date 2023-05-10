# coding : utf-8
# Author : yuxiang Zeng

import platform
import time

from torch.utils.data import Dataset, DataLoader

from datasets.RecEraser import interaction_based_balanced_parition
from datasets.SISA import random_split_to_shard, random_split_to_shard2
from datasets.data_generator import get_train_valid_test_dataset
from datasets.data_partition import eraser, eraser2, eraser3, eraser5, eraser4_2
from utility.utils import *
import numpy as np
import torch as t
import pickle as pk
import dgl as d
from tqdm import *


# 加载处理好的数据集
def load_data(args):
    string = ''

    if args.dataset == 'rt':
        string = args.path + 'rt.pk'

    if args.dataset == 'tp':
        string = args.path + 'tp.pk'

    data = pk.load(open(string, 'rb'))

    return data


# 合并异常值数据矩阵
def merge_Tensor_outlier(Tensor, outlier):
    merge = Tensor
    for i in range(len(Tensor)):
        for j in range(len(Tensor[0])):
            if Tensor[i][j] >= 0 and outlier[i][j] == 1:
                merge[i][j] = 0

    return merge


# 数据集定义
class ShardedTensorDataset(Dataset):

    def __getitem__(self, index):
        output = self.idx[index]
        userIdx, itemIdx, value = t.as_tensor(output[0]).long(), t.as_tensor(output[1]).long(), output[2]
        return userIdx, itemIdx, value

    def __init__(self, data, First, args):
        self.path = args.path
        self.args = args
        self.data = data
        self.data[self.data == -1] = 0
        self.idx = self.get_index(self.data)

        self.n_users, self.n_items = [], []
        self.sliceId = 0
        self.max_value = data.max()
        self.label = None

        if First:
            self.n_users, self.n_items = self.data.shape

            # 制作一定采样密度下的随机数据集
            self.train_Tensor, self.valid_Tensor, self.test_Tensor, self.max_value = get_train_valid_test_dataset(self.data, args)

            self.split_valid_Tensor, self.split_test_Tensor = [], []

            # 是否采用数据预处理
            if args.processed == 1:
                # 获得异常值矩阵
                string = ''
                if args.dataset == 'rt':
                    string = args.path + 'rt_outlier.pk'

                if args.dataset == 'tp':
                    string = args.path + 'tp_outlier.pk'

                outlier = pk.load(open(string, 'rb'))

                log('\t采用异常值处理')
                self.train_Tensor = merge_Tensor_outlier(self.train_Tensor, outlier)
                log('\t异常值处理完毕')

            # SISA
            if args.part_type == 1:
                self.split_train_Tensor = random_split_to_shard(self.train_Tensor, args)
                log('\tSISA 切割完毕')

            # SISA优化后
            if args.part_type == 2:
                datasets_ = self.train_Tensor, self.valid_Tensor, self.test_Tensor
                self.split_train_Tensor, self.split_valid_Tensor, self.split_test_Tensor, self.label = random_split_to_shard2(datasets_, args)
                log('\tSISA 切割完毕')

            # node2vec_Earser
            flag = True
            if args.part_type == 3:
                self.split_train_Tensor = interaction_based_balanced_parition(self.train_Tensor, args)
                log('\tReceraser 基于交互切割完毕')

            if args.part_type == 4:
                # 非平衡切割
                self.split_train_Tensor, self.label = eraser(self.train_Tensor, args)
                log('\tNode2Vec 基于用户聚类非均衡切割完毕')

            if args.part_type == 5:
                set_seed(args.random_state)
                datasets_ = self.train_Tensor, self.valid_Tensor, self.test_Tensor
                self.split_train_Tensor, self.split_valid_Tensor, self.split_test_Tensor, self.label = eraser4_2(datasets_, args)
                # log('\tNode2Vec 基于用户聚类均衡切割完毕')

            if args.part_type == 6:
                # 简单聚类
                set_seed(args.random_state)
                datasets_ = self.train_Tensor, self.valid_Tensor, self.test_Tensor
                self.split_train_Tensor, self.split_valid_Tensor, self.split_test_Tensor, self.label = eraser3(datasets_, args)
                log('\t地区特征 基于用户切割完毕')

            if args.part_type == 7:
                self.split_train_Tensor, self.label = eraser2(self.train_Tensor, args)
                log('\tNode2Vec 基于服务聚类非均衡切割完毕')

            if args.part_type == 8:
                set_seed(args.random_state)
                datasets_ = self.train_Tensor, self.valid_Tensor, self.test_Tensor
                self.split_train_Tensor, self.split_valid_Tensor, self.split_test_Tensor, self.label = eraser5(datasets_, args)
                log('\tNode2Vec 基于服务聚类均衡切割完毕')

            # retrain
            if args.part_type == 10:
                self.split_train_Tensor = []
                for i in range(args.slices):
                    self.split_train_Tensor += [self.train_Tensor]
                    self.split_valid_Tensor += [self.valid_Tensor]

                log('\tRetrain')

            ################################################################
            # 验证集与测试集
            if flag:  # 防止Eraser重复加
                for i in range(args.slices):
                    if args.part_type in [1, 3]:
                        self.split_valid_Tensor += [self.valid_Tensor]
                        self.split_test_Tensor += [self.test_Tensor]
    ##################################################################

    def setSliceId(self, sliceId):
        self.sliceId = sliceId
        self.idx = self.get_index(self.data)

    def __len__(self):
        return len(self.idx)

    @staticmethod
    def get_index(data):
        userIdx, itemIdx = data.nonzero()
        value = []
        for i in range(len(userIdx)):
            value.append(data[userIdx[i], itemIdx[i]])
        index = np.transpose([userIdx, itemIdx, np.array(value)])
        return t.tensor(index)

    # 返回该桶数据
    def get_tensor(self, sliceId):
        return self.split_train_Tensor[sliceId], self.split_valid_Tensor[sliceId], self.split_test_Tensor[sliceId]

    # 返回完整数据
    def full(self):
        return self.train_Tensor, self.valid_Tensor, self.test_Tensor


##################################################################
# GraphMF
def create_graph():

    userg = d.graph([])
    servg = d.graph([])
    user_lookup = FeatureLookup()
    serv_lookup = FeatureLookup()
    ufile = pd.read_csv('./datasets/data/WSDREAM/原始数据/userlist_table.csv')
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()
    ulines = ulines

    sfile = pd.read_csv('./datasets/data/WSDREAM/原始数据/wslist_table.csv')
    sfile = pd.DataFrame(sfile)
    slines = sfile.to_numpy()
    slines = slines

    for i in range(339):
        user_lookup.register('User', i)
    for j in range(5825):
        serv_lookup.register('Serv', j)
    for ure in ulines[:, 2]:
        user_lookup.register('URE', ure)
    for uas in ulines[:, 4]:
        user_lookup.register('UAS', uas)

    for sre in slines[:, 4]:
        serv_lookup.register('SRE', sre)
    for spr in slines[:, 2]:
        serv_lookup.register('SPR', spr)
    for sas in slines[:, 6]:
        serv_lookup.register('SAS', sas)

    userg.add_nodes(len(user_lookup))
    servg.add_nodes(len(serv_lookup))

    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[2])
        if not userg.has_edges_between(uid, ure):
            userg.add_edges(uid, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(uid, uas):
            userg.add_edges(uid, uas)

    for line in slines:
        sid = line[0]
        sre = serv_lookup.query_id(line[4])
        if not servg.has_edges_between(sid, sre):
            servg.add_edges(sid, sre)

        sas = serv_lookup.query_id(line[6])
        if not servg.has_edges_between(sid, sas):
            servg.add_edges(sid, sas)

        spr = serv_lookup.query_id(line[2])
        if not servg.has_edges_between(sid, spr):
            servg.add_edges(sid, spr)

    userg = d.add_self_loop(userg)
    userg = d.to_bidirected(userg)
    servg = d.add_self_loop(servg)
    servg = d.to_bidirected(servg)
    return user_lookup, serv_lookup, userg, servg


class FeatureLookup:

    def __init__(self):
        self.__inner_id_counter = 0
        self.__inner_bag = {}
        self.__category = set()
        self.__category_bags = {}
        self.__inverse_map = {}

    def register(self, category, value):
        # 添加进入类别
        self.__category.add(category)
        # 如果类别不存在若无则，则新增一个类别子树
        if category not in self.__category_bags:
            self.__category_bags[category] = {}

        # 如果值不在全局索引中，则创建之，id += 1
        if value not in self.__inner_bag:
            self.__inner_bag[value] = self.__inner_id_counter
            self.__inverse_map[self.__inner_id_counter] = value
            # 如果值不存在与类别子树，则创建之
            if value not in self.__category_bags[category]:
                self.__category_bags[category][value] = self.__inner_id_counter
            self.__inner_id_counter += 1

    def query_id(self, value):
        # 返回索引id
        return self.__inner_bag[value]

    def query_value(self, id):
        # 返回值
        return self.__inverse_map[id]

    def __len__(self):
        return len(self.__inner_bag)