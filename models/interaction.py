# coding : utf-8
# Author : yuxiang Zeng

from time import time
import torch as t
import numpy as np
import pandas as pd
from torch.nn import *
from torch.nn import functional as F
# from dgl.nn.pytorch import SAGEConv
import math


class CoSTCo(Module):

    def __init__(self, rank):
        super(CoSTCo, self).__init__()
        self.num_channels = 32
        self.conv1 = Sequential(LazyConv2d(self.num_channels, kernel_size=(3, 1)), ReLU())
        self.conv2 = Sequential(LazyConv2d(self.num_channels, kernel_size=(1, rank)), ReLU())
        self.flatten = Flatten()
        self.linear = Sequential(LazyLinear(rank), ReLU())
        self.output = Sequential(LazyLinear(1), Sigmoid())

    def forward(self, user_embeds, item_embeds, time_embeds):
        # stack as [batch, N, dim]
        x = t.stack([time_embeds, user_embeds, item_embeds], dim=1)

        # reshape to [batch, 1, N, dim]
        x = t.unsqueeze(x, dim=1)

        # conv
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x.flatten()


class TMLPBlock(Module):

    def __init__(self, in_feats, out_feats):
        super(TMLPBlock, self).__init__()

        self.A = Linear(in_feats, out_feats, bias=False)
        self.B = Linear(in_feats, out_feats, bias=False)
        self.C = Linear(in_feats, out_feats, bias=False)
        self.bias = Parameter(t.zeros((out_feats, out_feats, out_feats)))

    def forward(self, input):
        # input = [n, r, r, r]
        x = input
        x = t.einsum('nijk, ri->nrjk', x, self.A.weight)
        x = t.einsum('nijk, rj->nirk', x, self.B.weight)
        x = t.einsum('nijk, rk->nijr', x, self.C.weight)
        x += self.bias
        return x


class NTM(Module):

    def __init__(self, rank):
        super(NTM, self).__init__()

        self.outer = Sequential(
            TMLPBlock(rank, rank // 2),
            TMLPBlock(rank // 2, rank // 4)
        )

        self.output = Sequential(
            LazyLinear(1),
            Sigmoid()
        )

    def forward(self, user_embeds, item_embeds, time_embeds):
        gcp = user_embeds * item_embeds * time_embeds
        out_prod = t.einsum('ni, nj, nk-> nijk', user_embeds, item_embeds, time_embeds)
        out_prod = self.outer(out_prod)
        rep = t.cat([gcp, out_prod.flatten(start_dim=1, end_dim=-1)], dim=-1)
        y = self.output(rep)
        return y.flatten()


class NTC(Module):
    def __init__(self, rank):
        super(NTC, self).__init__()
        self.cnn = ModuleList()

        for i in range(int(math.log2(rank))):
            in_channels = 1 if i == 0 else rank
            conv_layer = Conv3d(in_channels=in_channels, out_channels=rank,
                                kernel_size=2, stride=2)
            self.cnn.append(conv_layer)
            self.cnn.append(LeakyReLU())

        self.score = Sequential(
            LazyLinear(1),
            Sigmoid()
        )

    def forward(self, user_embeds, item_embeds, time_embeds):

        outer_prod = t.einsum('ni, nj, nk-> nijk', user_embeds, item_embeds, time_embeds)
        outer_prod = t.unsqueeze(outer_prod, dim=1)
        rep = outer_prod
        for layer in self.cnn:
            rep = layer(rep)
        y = self.score(rep.squeeze())
        return y.flatten()


# 第一个模型
class User_embeds_NeuCF(t.nn.Module):
    def __init__(self, n_users, factor_num, args):
        super(User_embeds_NeuCF, self).__init__()
        self.num_layers = args.num_layers
        self.dimension_gmf = factor_num
        self.dimension_mlp = factor_num * (2 ** (self.num_layers - 1))
        self.embed_user_GMF = t.nn.Embedding(n_users, self.dimension_gmf)
        self.embed_user_MLP = t.nn.Embedding(n_users, self.dimension_mlp)

    def forward(self, UserIdx):
        user_embed = self.embed_user_GMF(UserIdx)
        embed_user_MLP = self.embed_user_MLP(UserIdx)

        return user_embed, embed_user_MLP


class Item_embeds_NeuCF(t.nn.Module):
    def __init__(self, n_services, factor_num, args):
        super(Item_embeds_NeuCF, self).__init__()
        self.num_layers = args.num_layers
        self.dimension_gmf = factor_num
        self.dimension_mlp = factor_num * (2 ** (self.num_layers - 1))
        self.embed_item_GMF = t.nn.Embedding(n_services, self.dimension_gmf)
        self.embed_item_MLP = t.nn.Embedding(n_services, self.dimension_mlp)

    def forward(self, itemIdx):
        item_embed = self.embed_item_GMF(itemIdx)
        embed_item_MLP = self.embed_item_MLP(itemIdx)
        return item_embed, embed_item_MLP


class NeuCF(t.nn.Module):
    def __init__(self, n_users, n_services, factor_num, args):
        super(NeuCF, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout

        MLP_modules = []
        for i in range(self.num_layers):
            input_size = factor_num * (2 ** (self.num_layers - i))
            MLP_modules.append(t.nn.Dropout(p=self.dropout))
            MLP_modules.append(t.nn.Linear(input_size, input_size // 2))
            MLP_modules.append(t.nn.ReLU())
        self.MLP_layers = t.nn.Sequential(*MLP_modules)

        self.predict_layer = t.nn.Linear(factor_num * 2, 1)

    def forward(self, user_embeds, item_embeds):
        user_embed, embed_user_MLP = user_embeds
        item_embed, embed_item_MLP = item_embeds

        gmf_output = user_embed * item_embed
        mlp_input = t.cat((embed_user_MLP, embed_item_MLP), -1)

        mlp_output = self.MLP_layers(mlp_input)

        prediction = self.predict_layer(t.cat((gmf_output, mlp_output), -1))

        return prediction.flatten()


#################################
# GraphMF
class NeuralCF(t.nn.Module):

    def __init__(self, input_size):
        super(NeuralCF, self).__init__()
        self.layers = t.nn.Sequential(
            t.nn.Linear(input_size, 128),
            t.nn.LayerNorm(128),
            t.nn.ReLU(),
            t.nn.Linear(128, 128),
            t.nn.LayerNorm(128),
            t.nn.ReLU(),
            t.nn.Linear(128, 1)
        )

    def forward(self, input):
        return self.layers(input)


from dgl.nn import SAGEConv


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
        return embeds


# 模型二
class NeuGraphMF(t.nn.Module):
    def __init__(self, usergraph, servgraph, args):
        super(NeuGraphMF, self).__init__()
        self.usergraph = usergraph
        self.servgraph = servgraph
        self.dim = args.dimension
        self.order = args.order
        self.ncf = NeuralCF(2 * args.dimension)

    def forward(self, user_embeds, serv_embeds):
        estimated = self.ncf(t.cat((user_embeds, serv_embeds), dim=-1)).sigmoid()
        estimated = estimated.reshape(user_embeds.shape[0])
        return estimated


# 第三个模型

class User_embed_CSMF(t.nn.Module):
    def __init__(self, n_users, args):
        super(User_embed_CSMF, self).__init__()
        self.UserList = pd.read_csv(args.path + 'userlist_idx.csv')
        self.device = 'cuda:0'

        self.UserEmbedding = t.nn.Embedding(n_users, args.dimension)
        self.UserASEmbedding = t.nn.Embedding(137, args.dimension)
        self.UserREEmbedding = t.nn.Embedding(31, args.dimension)

        for layer in self.children():
            if isinstance(layer, t.nn.Embedding):
                param_shape = layer.weight.shape
                layer.weight.data = t.from_numpy(np.random.uniform(-1, 1, size=param_shape))

    def forward(self, UserIdx):
        user = np.array(UserIdx.cpu(), dtype='int32')
        UserAS = t.tensor(np.array(self.UserList['[AS]'][user]))
        UserRE = t.tensor(np.array(self.UserList['[Country]'][user]))

        UserIdx = UserIdx.to(self.device)
        UserAS = UserAS.to(self.device)
        UserRE = UserRE.to(self.device)

        user_embed = self.UserEmbedding(UserIdx)
        uas_embed = self.UserASEmbedding(UserAS)
        ure_embed = self.UserREEmbedding(UserRE)

        user_vec = user_embed + uas_embed + ure_embed

        return user_vec


class Item_embed_CSMF(t.nn.Module):
    def __init__(self, n_items, args):
        super(Item_embed_CSMF, self).__init__()
        self.ServList = pd.read_csv(args.path + 'wslist_idx.csv')
        self.device = 'cuda:0'

        self.ServEmbedding = t.nn.Embedding(n_items, args.dimension)
        self.ServASEmbedding = t.nn.Embedding(1603, args.dimension)
        self.ServREEmbedding = t.nn.Embedding(74, args.dimension)
        self.ServPrEmbedding = t.nn.Embedding(2699, args.dimension)
        for layer in self.children():
            if isinstance(layer, t.nn.Embedding):
                param_shape = layer.weight.shape
                layer.weight.data = t.from_numpy(np.random.uniform(-1, 1, size=param_shape))

    def forward(self, ServIdx):
        serv = np.array(ServIdx.cpu(), dtype='int32')

        ServAS = t.tensor(np.array(self.ServList['[AS]'][serv]))
        ServRE = t.tensor(np.array(self.ServList['[Country]'][serv]))
        ServPr = t.tensor(np.array(self.ServList['[Service Provider]'][serv]))

        ServIdx = ServIdx.to(self.device)

        ServAS = ServAS.to(self.device)
        ServRE = ServRE.to(self.device)
        ServPr = ServPr.to(self.device)

        serv_embed = self.ServEmbedding(ServIdx)
        sas_embed = self.ServASEmbedding(ServAS)
        sre_embed = self.ServREEmbedding(ServRE)
        spr_embed = self.ServPrEmbedding(ServPr)
        serv_vec = serv_embed + sas_embed + sre_embed + spr_embed

        return serv_vec


class CSMF(t.nn.Module):

    def __init__(self, n_users, n_services, args):
        super(CSMF, self).__init__()

        self.device = 'cuda:0'
        self.user_norm = t.nn.LayerNorm(args.dimension)
        self.item_norm = t.nn.LayerNorm(args.dimension)
        self.norm = t.nn.LayerNorm(args.dimension)
        for layer in self.children():
            if isinstance(layer, t.nn.Embedding):
                param_shape = layer.weight.shape
                layer.weight.data = t.from_numpy(np.random.uniform(-1, 1, size=param_shape))
        self.predict_layer = t.nn.Linear(args.dimension, 1)

    def forward(self, user_vec, serv_vec):
        user_vec = user_vec.to(t.float32)
        serv_vec = serv_vec.to(t.float32)
        user_vec = self.user_norm(user_vec)
        serv_vec = self.item_norm(serv_vec)
        inner_prod = user_vec * serv_vec
        pred = self.norm(inner_prod.float()).sum(dim=-1).sigmoid()
        # pred = self.predict_layer(self.norm(inner_prod.float())).sigmoid()
        return pred.flatten()


# 第一个模型
class Pure_Mf_User(t.nn.Module):
    def __init__(self, n_users, factor_num, args):
        super(Pure_Mf_User, self).__init__()
        self.embed_user_GMF = t.nn.Embedding(n_users, factor_num)

    def forward(self, UserIdx):
        user_embed = self.embed_user_GMF(UserIdx)
        return user_embed


class Pure_Mf_Item(t.nn.Module):
    def __init__(self, n_services, factor_num, args):
        super(Pure_Mf_Item, self).__init__()
        self.embed_item_GMF = t.nn.Embedding(n_services, factor_num)

    def forward(self, itemIdx):
        item_embed = self.embed_item_GMF(itemIdx)
        return item_embed


class Pure_Mf(t.nn.Module):
    def __init__(self, args):
        super(Pure_Mf, self).__init__()
        self.predict_layer = t.nn.Linear(args.dimension, 1)

    def forward(self, user_embed, item_embed):
        gmf_output = user_embed * item_embed
        prediction = self.predict_layer(gmf_output)
        return prediction.flatten()


# 第五个模型
class Slice_user_embedding(Module):
    def __init__(self, n_users, args):
        super(Slice_user_embedding, self).__init__()
        self.embed_user_MLP = t.nn.Embedding(n_users, args.dimension)

    def forward(self, UserIdx):
        embed_user_MLP = self.embed_user_MLP(UserIdx)
        return embed_user_MLP


class Slice_item_embedding(Module):
    def __init__(self, n_services, args):
        super(Slice_item_embedding, self).__init__()
        self.embed_item_MLP = t.nn.Embedding(n_services, args.dimension)

    def forward(self, itemIdx):
        embed_item_MLP = self.embed_item_MLP(itemIdx)
        return embed_item_MLP


class MLP_MF(Module):
    def __init__(self, args):
        super(MLP_MF, self).__init__()
        self.dropout = args.dropout

        MLP_modules = []
        for i in range(self.num_layers):
            input_size = args.dimension * (2 ** (self.num_layers - i))
            MLP_modules.append(t.nn.Dropout(p=self.dropout))
            MLP_modules.append(t.nn.Linear(input_size, input_size // 2))
            MLP_modules.append(t.nn.ReLU())
        self.MLP_layers = t.nn.Sequential(*MLP_modules)
        self.predict_layer = t.nn.Linear(args.dimension, 1)

    def forward(self, embed_user_MLP, embed_item_MLP):
        mlp_input = t.cat((embed_user_MLP, embed_item_MLP), -1)
        mlp_output = self.MLP_layers(mlp_input)
        prediction = self.predict_layer(mlp_output)
        return prediction.flatten()
