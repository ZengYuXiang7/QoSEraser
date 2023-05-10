# coding : utf-8
# Author : yuxiang Zeng

import pickle as pk
import dgl as d
import pandas as pd
import numpy as np
from node2vec import Node2Vec

# node2vec_dim, node2vec_length, node2vec_walk, node2vec_epochs, node2vec_batchsize
# 128           5                50             20               32


# Node2vec get pretrain
def get_user_embedding(args):

    userg = d.graph([])
    user_lookup = FeatureLookup()
    ufile = pd.read_csv('./datasets/data/WSDREAM/原始数据/userlist_table.csv')
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()

    for i in range(339):
        user_lookup.register('User', i)

    for ure in ulines[:, 2]:
        user_lookup.register('URE', ure)

    for uas in ulines[:, 4]:
        user_lookup.register('UAS', uas)

    userg.add_nodes(len(user_lookup))

    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[2])

        if not userg.has_edges_between(uid, ure):
            userg.add_edges(uid, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(uid, uas):
            userg.add_edges(uid, uas)

    userg = d.add_self_loop(userg)
    userg = d.to_bidirected(userg)

    G = userg.to_networkx()

    # 设置node2vec参数
    node2vec = Node2Vec(
        G,
        dimensions=args.node2vec_dim,  # 嵌入维度
        p=1,  # 回家参数
        q=0.5,  # 外出参数
        walk_length=args.node2vec_length,  # 随机游走最大长度
        num_walks=args.node2vec_walk,  # 每个节点作为起始节点生成的随机游走个数
        workers=1,  # 并行线程数
        seed=args.random_state
    )

    # p=1, q=0.5, n_clusters=6。DFS深度优先搜索，挖掘同质社群
    # p=1, q=2, n_clusters=3。BFS宽度优先搜索，挖掘节点的结构功能。

    # 训练Node2Vec
    model = node2vec.fit(
        window=args.node2vec_windows,  # Skip-Gram窗口大小
        epochs=args.node2vec_epochs,
        min_count = 3,  # 忽略出现次数低于此阈值的节点（词）
        batch_words=args.node2vec_batchsize,  # 每个线程处理的数据量
        seed = args.random_state
    )
    ans = model.wv.vectors[:339]

    user_embedding = np.array(ans)

    pk.dump(user_embedding, open(f'./pretrain/user_embeds.pk', 'wb'))

    return user_embedding


def get_item_embedding(args):
    servg = d.graph([])

    serv_lookup = FeatureLookup()
    sfile = pd.read_csv('./datasets/data/WSDREAM/原始数据/wslist_table.csv')
    sfile = pd.DataFrame(sfile)
    slines = sfile.to_numpy()

    for i in range(5825):
        serv_lookup.register('Sid', i)

    for sre in slines[:, 4]:
        serv_lookup.register('SRE', sre)

    for spr in slines[:, 2]:
        serv_lookup.register('SPR', spr)

    for sas in slines[:, 6]:
        serv_lookup.register('SAS', sas)

    servg.add_nodes(len(serv_lookup))

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

    servg = d.add_self_loop(servg)
    servg = d.to_bidirected(servg)

    # node2vec
    from node2vec import Node2Vec
    G = servg.to_networkx()
    # 设置node2vec参数
    node2vec = Node2Vec(G,
                        dimensions=args.node2vec_dim,  # 嵌入维度
                        p=1,  # 回家参数
                        q=0.5,  # 外出参数
                        walk_length=args.node2vec_length,  # 随机游走最大长度
                        num_walks=args.node2vec_walk,  # 每个节点作为起始节点生成的随机游走个数
                        workers=1  # 并行线程数
                        )

    # p=1, q=0.5, n_clusters=6。DFS深度优先搜索，挖掘同质社群
    # p=1, q=2, n_clusters=3。BFS宽度优先搜索，挖掘节点的结构功能。

    # 训练Node2Vec，参数文档见 gensim.models.Word2Vec
    model = node2vec.fit(window=args.node2vec_windows,  # Skip-Gram窗口大小
                         epochs=args.node2vec_epochs,
                         min_count=1,  # 忽略出现次数低于此阈值的节点（词）
                         batch_words=args.node2vec_batchsize,  # 每个线程处理的数据量
                         seed = args.random_state
                         )

    ans = model.wv.vectors[:5825]

    item_embedding = np.array(ans)

    return item_embedding


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

    def query_value(self, idx):
        # 返回值
        return self.__inverse_map[idx]

    def __len__(self):
        return len(self.__inner_bag)
