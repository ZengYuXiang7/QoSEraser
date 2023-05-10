# coding : utf-8
# Author : yuxiang Zeng
import collections
import configparser
import logging

from dgl.nn.pytorch import SAGEConv
from torch.utils.data import Dataset, DataLoader

from utility.record import check_records
from utility.utils import *
import time
from tqdm import *
import pandas as pd
import pickle as pk
import numpy as np
import torch as t
import dgl as d

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def ErrMetrics(realVec, estiVec):
    realVec = t.as_tensor(realVec)
    estiVec = t.as_tensor(estiVec)

    absError = t.abs(estiVec - realVec)

    MAE = t.mean(absError)

    RMSE = t.linalg.norm(absError) / t.sqrt(t.tensor(absError.shape[0]))

    NMAE = MAE / (t.sum(realVec) / absError.shape[0])

    relativeError = absError / realVec

    MRE = t.tensor(np.percentile(relativeError, 50))  # Mean Relative Error
    NPRE = t.tensor(np.percentile(relativeError, 90))  #

    MAPE = t.mean(t.abs(t.div(absError, True)))
    MARE = t.div(t.sum(absError), t.sum(realVec))

    return MAE, RMSE, NMAE, MRE, NPRE


class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, args, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.args = args
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves models when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving models ...')
        self.val_loss_min = val_loss


# 第一个模型
class NeuCF(t.nn.Module):
    def __init__(self, n_users, n_services, args):
        super(NeuCF, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.dimension = args.dimension
        self.dimension_gmf = args.dimension
        self.dimension_mlp = args.dimension * (2 ** (self.num_layers - 1))
        self.embed_user_GMF = t.nn.Embedding(n_users, self.dimension_gmf)
        self.embed_user_MLP = t.nn.Embedding(n_users, self.dimension_mlp)
        self.embed_item_GMF = t.nn.Embedding(n_services, self.dimension_gmf)
        self.embed_item_MLP = t.nn.Embedding(n_services, self.dimension_mlp)

        MLP_modules = []
        for i in range(self.num_layers):
            input_size = self.dimension * (2 ** (self.num_layers - i))
            MLP_modules.append(t.nn.Dropout(p=self.dropout))
            MLP_modules.append(t.nn.Linear(input_size, input_size // 2))
            MLP_modules.append(t.nn.ReLU())
        self.MLP_layers = t.nn.Sequential(*MLP_modules)
        self.predict_layer = t.nn.Linear(self.dimension * 2, 1)

    def forward(self, UserIdx, itemIdx):
        user_embed = self.embed_user_GMF(UserIdx)
        embed_user_MLP = self.embed_user_MLP(UserIdx)

        item_embed = self.embed_item_GMF(itemIdx)
        embed_item_MLP = self.embed_item_MLP(itemIdx)

        gmf_output = user_embed * item_embed
        mlp_input = t.cat((embed_user_MLP, embed_item_MLP), -1)

        mlp_output = self.MLP_layers(mlp_input)

        prediction = self.predict_layer(t.cat((gmf_output, mlp_output), -1))

        return prediction.flatten()


class CSMF(t.nn.Module):

    def __init__(self, n_users, n_services, args):
        super(CSMF, self).__init__()
        self.UserList = pd.read_csv(args.path + 'userlist_idx.csv')
        self.ServList = pd.read_csv(args.path + 'wslist_idx.csv')

        self.UserEmbedding = t.nn.Embedding(n_users, args.dimension)
        self.UserASEmbedding = t.nn.Embedding(137, args.dimension)
        self.UserREEmbedding = t.nn.Embedding(31, args.dimension)
        self.ServEmbedding = t.nn.Embedding(n_services, args.dimension)
        self.ServASEmbedding = t.nn.Embedding(1603, args.dimension)
        self.ServREEmbedding = t.nn.Embedding(74, args.dimension)
        self.ServPrEmbedding = t.nn.Embedding(2699, args.dimension)

        self.user_norm = t.nn.LayerNorm(args.dimension)
        self.serv_norm = t.nn.LayerNorm(args.dimension)

        for layer in self.children():
            if isinstance(layer, t.nn.Embedding):
                param_shape = layer.weight.shape
                layer.weight.data = t.from_numpy(np.random.uniform(-1, 1, size=param_shape))
        self.device = 'cuda:0'

        self.norm = t.nn.LayerNorm(args.dimension)

    def forward(self, UserIdx, ServIdx):
        user = np.array(UserIdx.cpu(), dtype='int32')
        UserAS = t.tensor(np.array(self.UserList['[AS]'][user]))
        UserRE = t.tensor(np.array(self.UserList['[Country]'][user]))
        UserIdx = UserIdx.to(self.device)
        UserAS = UserAS.to(self.device)
        UserRE = UserRE.to(self.device)
        user_embed = self.UserEmbedding(UserIdx)
        uas_embed = self.UserASEmbedding(UserAS)
        ure_embed = self.UserREEmbedding(UserRE)
        serv = np.array(ServIdx.cpu(), dtype='int32')
        ServAS = t.tensor(np.array(self.ServList['[AS]'][serv]))
        ServRE = t.tensor(np.array(self.ServList['[Country]'][serv]))
        ServPr = t.tensor(np.array(self.ServList['[Service Provider]'][serv]))
        user_vec = user_embed + uas_embed + ure_embed
        ServIdx = ServIdx.to(self.device)
        ServAS = ServAS.to(self.device)
        ServRE = ServRE.to(self.device)
        ServPr = ServPr.to(self.device)
        serv_embed = self.ServEmbedding(ServIdx)
        sas_embed = self.ServASEmbedding(ServAS)
        sre_embed = self.ServREEmbedding(ServRE)
        spr_embed = self.ServPrEmbedding(ServPr)
        serv_vec = serv_embed + sas_embed + sre_embed + spr_embed
        user_vec = self.user_norm(user_vec.to(t.float32))
        serv_vec = self.serv_norm(serv_vec.to(t.float32))
        inner_prod = user_vec * serv_vec
        inner_prod = self.norm(inner_prod.float())
        tmp = t.sum(inner_prod, -1)
        pred = tmp.sigmoid()
        return pred.flatten()


class Pure_Mf(t.nn.Module):
    def __init__(self, n_users, n_services, args):
        super(Pure_Mf, self).__init__()
        self.embed_user_GMF = t.nn.Embedding(n_users, args.dimension)
        self.embed_item_GMF = t.nn.Embedding(n_services, args.dimension)
        self.predict_layer = t.nn.Linear(args.dimension, 1)

    def forward(self, UserIdx, itemIdx):
        user_embed = self.embed_user_GMF(UserIdx)
        item_embed = self.embed_item_GMF(itemIdx)
        gmf_output = user_embed * item_embed
        prediction = self.predict_layer(gmf_output)
        return prediction.flatten()


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


class GraphSAGEConv(t.nn.Module):

    def __init__(self, graph, dim, order=3):
        super(GraphSAGEConv, self).__init__()
        self.order = order
        self.graph = graph
        self.embedding = t.nn.Parameter(t.Tensor(self.graph.number_of_nodes(), dim))
        t.nn.init.kaiming_normal_(self.embedding)
        self.graph.ndata['L0'] = self.embedding
        self.layers = t.nn.ModuleList([SAGEConv(dim, dim, aggregator_type='gcn') for _ in range(order)])
        self.norms = t.nn.ModuleList([t.nn.LayerNorm(dim) for _ in range(order)])
        self.acts = t.nn.ModuleList([t.nn.ELU() for _ in range(order)])

    def forward(self, uid):
        g = self.graph.to('cuda')
        feats = g.ndata['L0']
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats).squeeze()
            feats = norm(feats)
            feats = act(feats)
            g.ndata[f'L{i + 1}'] = feats

        embeds = g.ndata[f'L{self.order}'][uid]
        # print(embeds.shape)
        return embeds


# 模型二
class NeuGraphMF(t.nn.Module):
    def __init__(self, usergraph, servgraph, args):
        super(NeuGraphMF, self).__init__()
        self.usergraph = usergraph
        self.servgraph = servgraph
        self.dim = args.dimension
        self.order = args.order
        self.user_embeds = GraphSAGEConv(self.usergraph, args.dimension, args.order)
        self.item_embeds = GraphSAGEConv(self.servgraph, args.dimension, args.order)
        self.layers = t.nn.Sequential(
            t.nn.Linear(2 * args.dimension, 128),
            # t.nn.Linear(2 * (args.order + 1) * args.dimension, 128),
            t.nn.LayerNorm(128),
            t.nn.ReLU(),
            t.nn.Linear(128, 128),
            t.nn.LayerNorm(128),
            t.nn.ReLU(),
            t.nn.Linear(128, 1)
        )

    def forward(self, userIdx, itemIdx):
        user_embeds = self.user_embeds(userIdx)
        serv_embeds = self.item_embeds(itemIdx)
        user_embeds = user_embeds.to(t.float32)
        serv_embeds = serv_embeds.to(t.float32)
        # print(user_embeds.shape, serv_embeds.shape,)
        estimated = self.layers(t.cat((user_embeds, serv_embeds), dim=-1)).sigmoid()
        estimated = estimated.reshape(user_embeds.shape[0])
        return estimated


def get_train_valid_test_dataset(tensor, args):
    quantile = np.percentile(tensor, q=100)
    tensor[tensor > quantile] = 0
    # tensor /= quantile
    density = args.density

    mask = np.random.rand(*tensor.shape).astype('float32')  # [0, 1]

    mask[mask > density] = 1
    mask[mask < density] = 0

    train_Tensor = tensor * (1 - mask)

    # size = int(0.00 * np.prod(tensor.shape))
    size = 0

    trIdx, tcIdx = mask.nonzero()
    p = np.random.permutation(len(trIdx))
    trIdx, tcIdx = trIdx[p], tcIdx[p]

    vrIdx, vcIdx = trIdx[:size], tcIdx[:size]
    trIdx, tcIdx = trIdx[size:], tcIdx[size:]

    valid_Tensor = np.zeros_like(tensor)
    test_Tensor = np.zeros_like(tensor)

    valid_Tensor[vrIdx, vcIdx] = tensor[vrIdx, vcIdx]
    test_Tensor[trIdx, tcIdx] = tensor[trIdx, tcIdx]

    return train_Tensor, valid_Tensor, test_Tensor, quantile


def load_data(args):
    string = ''

    if args.dataset == 'rt':
        string = args.path + 'rt.pk'

    if args.dataset == 'tp':
        string = args.path + 'tp.pk'

    data = pk.load(open(string, 'rb'))

    return data


class QoSDataset(Dataset):
    def __getitem__(self, index):
        output = self.idx[index]
        userIdx, itemIdx, value = t.as_tensor(output[0]).long(), t.as_tensor(output[1]).long(), output[2]
        return userIdx, itemIdx, value

    def __len__(self):
        return len(self.idx)

    def __init__(self, data, args):
        self.path = args.path
        self.args = args
        self.data = data
        self.data[self.data == -1] = 0
        self.idx = self.get_index(self.data)
        self.max_value = data.max()
        self.train_Tensor, self.valid_Tensor, self.test_Tensor, self.max_value = get_train_valid_test_dataset(self.data,
                                                                                                              args)

    @staticmethod
    def get_index(data):
        userIdx, itemIdx = data.nonzero()
        value = []
        for i in range(len(userIdx)):
            value.append(data[userIdx[i], itemIdx[i]])
        index = np.transpose([userIdx, itemIdx, np.array(value)])
        return t.tensor(index)

    def get_tensor(self):
        return self.train_Tensor, self.valid_Tensor, self.test_Tensor


def get_dataloaders(dataset, args):
    train, valid, test = dataset.get_tensor()
    train_set = QoSDataset(train, args)
    valid_set = QoSDataset(valid, args)
    test_set = QoSDataset(test, args)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True,
                              num_workers=8, prefetch_factor=4)
    valid_loader = DataLoader(valid_set, batch_size=131072, drop_last=False, shuffle=False, pin_memory=True,
                              num_workers=8, prefetch_factor=4)
    test_loader = DataLoader(test_set, batch_size=131072, drop_last=False, shuffle=False, pin_memory=True,
                             num_workers=8, prefetch_factor=4)

    return train_loader, valid_loader, test_loader


# 训练函数
def train_slice(model, train_loader, args):
    training_time = []
    loss_function = t.nn.L1Loss()

    lr = 1e-3 if args.interaction != 'GraphMF' else 8e-3
    optimizer_model = t.optim.AdamW(model.parameters(), lr=lr) if args.interaction != 'GraphMF' else t.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-3)
    model = model.cuda()
    loss_function = loss_function.cuda()
    # 正式训练部分
    best_epoch = 0
    for epoch in trange(args.slice_epochs):
        if args.interaction == 'GraphMF':
            if epoch % 5 == 0:
                lr /= 2
                optimizer_model = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.slice_decay)
        t.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in train_loader:
            userIdx, itemIdx, mVal = train_Batch
            if args.devices == 'gpu':
                userIdx, itemIdx, mVal = userIdx.cuda(), itemIdx.cuda(), mVal.cuda()
            pred = model.forward(userIdx, itemIdx)
            optimizer_model.zero_grad()
            loss = loss_function(pred, mVal)
            loss.backward()
            optimizer_model.step()
        t2 = time.time()
        training_time.append(t2 - t1)
        t.set_grad_enabled(False)
    sum_time = sum(training_time)
    return sum_time


###################################################################################################################################

# 每轮实验的进行
def run(round, args):
    ###################################################################################################################################
    # 数据模型读取
    df = np.array(load_data(args))
    dataset = QoSDataset(df, args)
    model = None
    if args.interaction == 'NeuCF':
        model = NeuCF(339, 5825, args)
    elif args.interaction == 'CSMF':
        model = CSMF(339, 5825, args)
    elif args.interaction == 'MF':
        model = Pure_Mf(339, 5825, args)
    elif args.interaction == 'GraphMF':
        user_lookup, serv_lookup, userg, servg = create_graph()
        model = NeuGraphMF(userg, servg, args)
    ###################################################################################################################################
    train_loader, valid_loader, test_loader = get_dataloaders(dataset, args)
    training_time = train_slice(model, train_loader, args)
    return training_time


###################################################################################################################################
# 执行程序
def main(args, start_round):
    log(str(args))
    ###################################################################################################################################
    config = configparser.ConfigParser()
    config.read('./utility/WSDREAM.conf')
    args = set_settings(args, config)
    ###################################################################################################################################
    results = collections.defaultdict(list)
    speed1 = {'NeuCF': 50, 'CSMF': 30, 'MF': 50, 'GraphMF': 30}
    speed2 = {'NeuCF': 50, 'CSMF': 30, 'MF': 60, 'GraphMF': 40}
    for dataset in ['rt', 'tp']:
        for inter in ['NeuCF', 'CSMF', 'MF', 'GraphMF']:
            args.interaction = inter
            for round in range(args.rounds):
                if dataset == 'rt':
                    args.slice_epochs = speed1[inter]
                else:
                    args.slice_epochs = speed2[inter]
                elapsed = run(round + 1, args)
                results[inter + dataset].append(elapsed)
        for inter in ['NeuCF', 'CSMF', 'MF', 'GraphMF']:
            print(f'{dataset}-----Inference Time: {inter}-{np.mean(results[inter + dataset]):.2f}s')

    print(results)
###################################################################################################################################


###################################################################################################################################
# 主程序
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # 实验常用
    parser.add_argument('--path', nargs='?', default='./datasets/data/WSDREAM/')
    parser.add_argument('--interaction', type=str, default='NeuCF')
    parser.add_argument('--dataset', type=str, default='rt')

    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--slice_epochs', type=int, default=100)
    parser.add_argument('--agg_epochs', type=int, default=100)
    parser.add_argument('--slice_lr', type=float, default=1e-3)
    parser.add_argument('--slice_decay', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--devices', type=str, default='gpu')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--slice', type=int, default=1)
    parser.add_argument('--ex', type=str, default='performance')

    parser.add_argument('--speed', type=list, default=[50, 30, 50, 30])
    parser.add_argument('--speed2', type=list, default=[50, 30, 60, 40])
    # 超参数
    parser.add_argument('--processed', type=int, default=0)  # 数据预处理

    # 切片
    parser.add_argument('--retrain', type=int, default=1)
    parser.add_argument('--slices', type=int, default=1)
    parser.add_argument('--part_type', type=int, default=10)  # 切割方法

    # NeuCF
    parser.add_argument('--density', type=float, default=0.1)  # 采样率
    parser.add_argument('--dropout', type=float, default=0.1)

    # NeuGraphMF, CSMF
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--random_state', type=int, default=1001)
    ###################################################################################################################################
    args = parser.parse_args()

    log('Experiment start!')
    main(args, 0)
    log('Experiment success!\n')

###################################################################################################################################
###################################################################################################################################
