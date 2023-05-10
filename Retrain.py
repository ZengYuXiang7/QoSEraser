# coding : utf-8
# Author : yuxiang Zeng
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

    def __init__(self, args, patience=7, verbose = False, delta=0):
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
    tensor /= quantile
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
        self.train_Tensor, self.valid_Tensor, self.test_Tensor, self.max_value = get_train_valid_test_dataset(self.data, args)

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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=4)
    valid_loader = DataLoader(valid_set, batch_size=131072, drop_last=False, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4)
    test_loader = DataLoader(test_set, batch_size=131072, drop_last=False, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=4)

    return train_loader, valid_loader, test_loader


# 训练函数
def train_slice(model, train_loader, valid_loader, args):
    training_time = []
    loss_function = t.nn.L1Loss()

    lr = 1e-3 if args.interaction != 'GraphMF' else 8e-3
    optimizer_model = t.optim.AdamW(model.parameters(), lr=lr) if args.interaction != 'GraphMF' else t.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    model = model.cuda()
    max_value = model.max_value
    loss_function = loss_function.cuda()

    early_stop = EarlyStopping(args, patience=10, verbose=False, delta=0)

    # 正式训练部分
    best_epoch = 0
    validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = 1e5, 1e5, 1e5, 1e5, 1e5
    for epoch in range(args.slice_epochs):
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

        # 设定为评估状态
        t.set_grad_enabled(False)

        reals, preds = [], []
        for valid_Batch in valid_loader:
            userIdx, itemIdx, mVal = valid_Batch
            if args.devices == 'gpu':
                userIdx, itemIdx, mVal = userIdx.cuda(), itemIdx.cuda(), mVal.cuda()
            pred = model.forward(userIdx, itemIdx)
            reals += mVal.tolist()
            preds += pred.tolist()

        reals = np.array(reals)
        preds = np.array(preds)

        # 大矩阵把整个
        validMAE, validRMSE, validNMAE, validMRE, validNPRE = ErrMetrics(reals * max_value, preds * max_value)

        log(f'Epoch {(epoch + 1):2d} : MAE : {validMAE:5.4f}  RMSE : {validRMSE:5.4f}  NMAE : {validNMAE:5.4f}  MRE : {validMRE:5.4f}  NPRE : {validNPRE:5.4f}')
        per_epoch_in_txt(args, epoch + 1, validMAE, validRMSE, validNMAE, validMRE, validNPRE, t2 - t1, False)

        if validMAE < validBestMAE:
            validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = validMAE, validRMSE, validNMAE, validMRE, validNPRE
            best_epoch = epoch + 1
            t.save(model.state_dict(), f'./pretrain/{args.dataset}_slices_{args.slices}_round_{args.part_type}_density_{args.density:.2f}_model_parameter.pkl')

        # early_stop(validMAE, model)
        if early_stop.early_stop:
            break
    sum_time = sum(training_time[: best_epoch])
    log(f'Best epoch {best_epoch :2d} : MAE : {validBestMAE:5.4f}  RMSE : {validBestRMSE:5.4f}  NMAE : {validBestNMAE:5.4f}  MRE : {validBestMRE:5.4f}  NPRE : {validBestNPRE:5.4f}')

    return validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE, sum_time


###################################################################################################################################


###################################################################################################################################
# 测试函数
def test_slice(model, test_loader, args):
    log('\t Retrain 测试模型')
    reals, preds = [], []
    # 开始测试
    for testBatch in tqdm(test_loader):
        userIdx, itemIdx, mVal = testBatch
        if args.devices == 'gpu':
            userIdx, itemIdx = userIdx.cuda(), itemIdx.cuda()
        pred = model.forward(userIdx, itemIdx)

        reals += mVal.tolist()
        preds += pred.tolist()

    reals = np.array(reals)
    preds = np.array(preds)
    testMAE, testRMSE, testNMAE, testMRE, testNPRE = ErrMetrics(reals * model.max_value, preds * model.max_value)

    log(f'Result : MAE : {testMAE:5.4f}  RMSE : {testRMSE:5.4f}  NMAE : {testNMAE:5.4f}  MRE : {testMRE:5.4f}  NPRE : {testNPRE:5.4f}')

    return testMAE, testRMSE, testNMAE, testMRE, testNPRE


###################################################################################################################################


# 每轮实验的进行
def run(round, logger, args):
    MAE, RMSE, NMAE, MRE, NPRE = None, None, None, None, None
    ###################################################################################################################################
    # 数据模型读取
    df = np.array(load_data(args))
    log('\t原始数据集读取完毕')
    dataset = QoSDataset(df, args)

    log('\t分切数据执行完毕')
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
    model.max_value = dataset.max_value
    log('\t模型加载完毕')
    ###################################################################################################################################
    train_loader, valid_loader, test_loader = get_dataloaders(dataset, args)
    MAE, RMSE, NMAE, MRE, NPRE, training_time = train_slice(model, train_loader, test_loader, args)
    model.load_state_dict(t.load(f'./pretrain/{args.dataset}_slices_{args.slices}_round_{args.part_type}_density_{args.density:.2f}_model_parameter.pkl'))
    MAE, RMSE, NMAE, MRE, NPRE = test_slice(model, test_loader, args)
    per_round_result_in_csv(args, round + 1, MAE, RMSE, NMAE, MRE, NPRE)
    per_round_result_in_txt(args, round + 1, MAE, RMSE, NMAE, MRE, NPRE), print('')
    per_slice_time_in_csv(args, round, 1, training_time)
    per_slice_time_in_txt(args, round, 1, training_time)
    log(f'Round {round:2d} : MAE = {MAE:.4f}, RMSE = {RMSE:.4f}, NMAE = {NMAE:.4f}, MRE = {MRE:.4f}, NPRE = {NPRE:.4f} train_time = {training_time:.2f} s')
    logger.info(f'Round {round:2d} : MAE = {MAE:.4f}, RMSE = {RMSE:.4f}, NMAE = {NMAE:.4f}, MRE = {MRE:.4f}, NPRE = {NPRE:.4f} train_time = {training_time:.2f} s\n')

    return MAE, RMSE, NMAE, MRE, NPRE


###################################################################################################################################
# 执行程序
def main(args, start_round):
    log(str(args))
    ###################################################################################################################################
    config = configparser.ConfigParser()
    config.read('./utility/WSDREAM.conf')
    args = set_settings(args, config)
    ###################################################################################################################################

    ###################################################################################################################################
    # 日志记录
    file = './Result/日志/' + args.ex + time.strftime('/%Y-%m-%d %H-%M-%S_', time.localtime(time.time())) + f'{args.dataset}_{args.interaction}_{args.density:.2f}.log'
    makedir(file[:file.find('2023') + 10])
    logging.basicConfig(level=logging.INFO, filename=file, filemode='w')
    logger = logging.getLogger('QoS-Unlearning')

    logger.info(f'Dataset : {args.dataset}     interaction : {args.interaction}')
    logger.info(f'Density : {(args.density * 100):.2f}%, epochs : {args.slice_epochs}')
    log(f'Dataset : {args.dataset}     interaction : {args.interaction}')
    log(f'Density : {(args.density * 100):.2f}%, epochs : {args.slice_epochs}')
    ###################################################################################################################################

    RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE = [], [], [], [], []

    for round in range(args.rounds):
        log(f'\nRound ' + str(round + 1) + ' experiment start!')
        MAE, RMSE, NMAE, MRE, NPRE = run(round + 1, logger, args)
        RunMAE += [MAE]
        RunRMSE += [RMSE]
        RunNMAE += [NMAE]
        RunMRE += [MRE]
        RunNPRE += [NPRE]
        if args.debug:
            per_round_result_in_csv(args, round + 1, MAE, RMSE, NMAE, MRE, NPRE)
            per_round_result_in_txt(args, round + 1, MAE, RMSE, NMAE, MRE, NPRE), print('')

    print('-' * 120)
    log(f'Dataset : {args.dataset}     interaction : {args.interaction}')
    log(f'Density : {(args.density * 100):.2f}%, slice_epochs : {args.slice_epochs}')
    log(f'Part_type : {args.part_type},    slices : {args.slices},       devices : {args.devices}\n')

    if args.rounds != 1:
        for round in range(args.rounds):
            log(f'RoundID {round + 1:} : MAE = {RunMAE[round] :.3f}, RMSE = {RunRMSE[round] :.3f}, NMAE = {RunNMAE[round] :.3f}, MRE = {RunMRE[round] :.3f}, NPRE = {RunNPRE[round] :.3f}')

    log(f'\nDensity {(args.density * 100):.2f}% : MAE = {np.mean(RunMAE, axis=0) :.3f}, RMSE = {np.mean(RunRMSE, axis=0) :.3f}, NMAE = {np.mean(RunNMAE, axis=0) :.3f}, MRE = {np.mean(RunMRE, axis=0) :.3f}, NPRE = {np.mean(RunNPRE, axis=0) :.3f}\n')
    logger.info(f'Density {(args.density * 100):.2f}% : MAE = {np.mean(RunMAE, axis=0) :.3f}, RMSE = {np.mean(RunRMSE, axis=0) :.3f}, NMAE = {np.mean(RunNMAE, axis=0) :.3f}, MRE = {np.mean(RunMRE, axis=0) :.3f}, NPRE = {np.mean(RunNPRE, axis=0) :.3f}')

    if args.record:
        if start_round != args.rounds + 1:
            final_result_in_txt(args, RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE)
            final_result_in_csv(args, RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE)

###################################################################################################################################


###################################################################################################################################
# 主程序
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # 实验常用
    parser.add_argument('--path', nargs='?', default='./datasets/data/WSDREAM/')
    parser.add_argument('--dataset', type=str, default='rt')
    parser.add_argument('--interaction', type=str, default='NeuCF')

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

    flag, start_round = False, 0

    if args.record:
        flag, start_round = check_records(args)

    # 自适应中断实验
    if not flag:
        # 是否存储结果
        if args.record and not args.debug:
            per_epoch_result_start(args)
            per_round_result_start(args)
            per_round_result_start_csv(args)
            per_slice_time_start(args)
            per_round_time_start_csv(args)
        log('Experiment start!')
        main(args, start_round)
    else:
        if start_round != args.rounds + 1:
            log('Continue experiment')
        else:
            log('All the experiments have been done!')
        main(args, start_round)

    log('Experiment success!\n')

###################################################################################################################################
###################################################################################################################################
