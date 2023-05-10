# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import torch as t
from torch.nn import *
import numpy as np
from datasets.dataset import create_graph
from models.interaction import *
from models.aggregator import UserAggregator, ItemAggregator, FinalMLP
from models.transformer import ScaledDotProductAttention
import pickle as pk
from utility.utils import log, makedir
from tqdm import *


# 选择用什么模型
def get_interaction_function(userg, servg, args):
    # 张量模型
    # if args.interaction == 'CoSTCo':
    #     return CoSTCo(max(args.ranks))
    #
    # if args.interaction == 'NTC':
    #     return NTC(max(args.ranks))
    #
    # if args.interaction == 'NTM':
    #     return NTM(max(args.ranks))
    #
    # if args.interaction == 'Inner':
    #     return Inner(max(args.ranks))

    # 模型一
    if args.interaction == 'NeuCF':
        return NeuCF(339, 5825, args.dimension, args).cuda()

    # 模型二
    if args.interaction == 'GraphMF':
        return NeuGraphMF(userg, servg, args).cuda()

    # 模型三,
    if args.interaction == 'CSMF':
        return CSMF(339, 5825, args).cuda()

    # 模型四,
    if args.interaction == 'MF':
        return Pure_Mf(args).cuda()


# 张量擦除模型
class TensorEraser(Module):

    def __init__(self, n_users, n_items, datasets, data, args):
        super(TensorEraser, self).__init__()
        self.n_users = n_users
        self.n_items = n_items

        # pytorch datasets
        self.dataset = datasets

        self.max_value = 0

        self.data = data
        self.args = args
        self.sliceId = 0
        self.sliceUserEmbeddings = ModuleList()
        self.sliceItemEmbeddings = ModuleList()
        self.slicedInterFunction = ModuleList()

        self.per_slice_user = None
        self.per_slice_item = None
        self.label = None

        # NeuGraphMF
        self.userg, self.servg = None, None
        if args.interaction == 'GraphMF':
            log('\tNeuGraphMF建图中')
            user_lookup, serv_lookup, userg, servg = create_graph()
            self.userg = userg
            self.servg = servg
            log('\tNeuGraphMF建图完毕')

        # 往每个切片里面塞入 嵌入向量模型 和 预测模型
        for _ in range(args.slices):
            user_embeds, item_embeds = None, None
            interaction = None

            if args.interaction == 'NeuCF':
                user_embeds = User_embeds_NeuCF(n_users, args.dimension, args)
                item_embeds = Item_embeds_NeuCF(n_items, args.dimension, args)
                interaction = get_interaction_function(n_users, n_items, args)

            elif args.interaction == 'GraphMF':
                user_embeds = GraphSAGEConv(self.userg, args.dimension, args.order)
                item_embeds = GraphSAGEConv(self.servg, args.dimension, args.order)
                interaction = get_interaction_function(self.userg, self.servg, args)

            elif args.interaction == 'CSMF':
                user_embeds = User_embed_CSMF(n_users, args)
                item_embeds = Item_embed_CSMF(n_items, args)
                interaction = get_interaction_function(n_users, n_items, args)

            elif args.interaction == 'MF':
                user_embeds = Pure_Mf_User(n_users, args.dimension, args)
                item_embeds = Pure_Mf_Item(n_items, args.dimension, args)
                interaction = get_interaction_function(n_users, n_items, args)

            if args.devices == 'gpu' and user_embeds:
                user_embeds, item_embeds = user_embeds.cuda(), item_embeds.cuda()

            # append these modules
            self.sliceUserEmbeddings += [user_embeds]
            self.sliceItemEmbeddings += [item_embeds]
            self.slicedInterFunction += [interaction]

        if args.devices == 'gpu':
            self.sliceUserEmbeddings, self.sliceItemEmbeddings = self.sliceUserEmbeddings.cuda(), self.sliceItemEmbeddings.cuda()
            self.slicedInterFunction = self.slicedInterFunction.cuda()

        # 最终全部模型的
        self.user_sliced_embeds_final = None
        self.item_sliced_embeds_final = None

        # NeuCF
        self.user_sliced_embeds_final_1 = []
        self.item_sliced_embeds_final_1 = []
        self.user_sliced_embeds_final_2 = []
        self.item_sliced_embeds_final_2 = []

        # 定义聚合模型
        if args.interaction == 'NeuCF':
            self.user_aggregator_1 = UserAggregator(self.dataset, args.dimension, args)
            self.item_aggregator_1 = ItemAggregator(self.dataset, args.dimension, args)

            self.user_aggregator_2 = UserAggregator(self.dataset, args.dimension * 2, args)
            self.item_aggregator_2 = ItemAggregator(self.dataset, args.dimension * 2, args)

        else:  # BasicMF CSMF graphMF
            self.user_aggregator = UserAggregator(self.dataset, args.dimension, args)
            self.item_aggregator = ItemAggregator(self.dataset, args.dimension, args)

        self.global_interaction = get_interaction_function(n_users, n_items, args)

        self.cache = {}
        self.slice_cache = [{} for _ in range(args.slices)]
        if args.devices == 'gpu':
            self.cuda()

    # 设置这个切片的标号
    def setSliceId(self, sliceId):
        self.sliceId = sliceId

    ###################################################################################################################################
    # 训练切片模型 前馈传播
    def train_slice_model(self, user, item):

        if self.args.devices == 'gpu':
            user, item = t.as_tensor(user.detach()).long(), t.as_tensor(item.detach()).long()

        # NCF CSMF GraphMF MF
        user_embeds = self.sliceUserEmbeddings[self.sliceId](user)
        item_embeds = self.sliceItemEmbeddings[self.sliceId](item)
        estimated = self.slicedInterFunction[self.sliceId](user_embeds, item_embeds)

        return estimated

    def prepare_test_slice_model(self):
        user_embeds = self.sliceUserEmbeddings[self.sliceId](t.tensor(range(339)).cuda())
        item_embeds = self.sliceItemEmbeddings[self.sliceId](t.tensor(range(5825)).cuda())
        if self.args.interaction == 'NeuCF':
            user_embeds_1, user_embeds_2 = self.sliceUserEmbeddings[self.sliceId](t.tensor(range(339)).cuda())
            item_embeds_1, item_embeds_2 = self.sliceItemEmbeddings[self.sliceId](t.tensor(range(5825)).cuda())
            self.slice_cache[self.sliceId]['user1'] = user_embeds_1
            self.slice_cache[self.sliceId]['user2'] = user_embeds_2
            self.slice_cache[self.sliceId]['item1'] = item_embeds_1
            self.slice_cache[self.sliceId]['item2'] = item_embeds_2
        elif self.args.interaction in ['CSMF', 'MF', 'GraphMF']:
            self.slice_cache[self.sliceId]['user'] = user_embeds
            self.slice_cache[self.sliceId]['item'] = item_embeds

    # 测试切片模型
    def test_slice_model(self, user, item):
        user, item = t.as_tensor(user.detach()).long(), t.as_tensor(item.detach()).long()
        user_embeds, item_embeds = None, None
        if self.args.interaction == 'NeuCF':
            user_embeds_1 = self.slice_cache[self.sliceId]['user1'][user]
            user_embeds_2 = self.slice_cache[self.sliceId]['user2'][user]
            item_embeds_1 = self.slice_cache[self.sliceId]['item1'][item]
            item_embeds_2 = self.slice_cache[self.sliceId]['item2'][item]
            user_embeds = user_embeds_1, user_embeds_2
            item_embeds = item_embeds_1, item_embeds_2
        elif self.args.interaction in ['CSMF', 'MF', 'GraphMF']:
            user_embeds = self.slice_cache[self.sliceId]['user'][user]
            item_embeds = self.slice_cache[self.sliceId]['item'][item]

        estimated = self.slicedInterFunction[self.sliceId](user_embeds, item_embeds)

        estimated[estimated < 0] = 0

        return estimated

    ###################################################################################################################################

    ###################################################################################################################################
    # 提前准备聚合，让所有的训练参数都保存下来
    def prepare_for_aggregation(self, args):

        if self.args.devices == 'gpu':
            userIdx = t.arange(self.n_users).cuda()
            itemIdx = t.arange(self.n_items).cuda()
        else:
            userIdx = t.arange(self.n_users)
            itemIdx = t.arange(self.n_items)

        # 已经对所有切片做了聚合了
        if self.args.interaction == 'NeuCF':
            user_sliced_embeds_final_1, user_sliced_embeds_final_2 = [], []
            item_sliced_embeds_final_1, item_sliced_embeds_final_2 = [], []
            for user_embeds in self.sliceUserEmbeddings:
                user_embeds_gmf, user_embeds_mlp = user_embeds(userIdx)
                user_embeds_gmf, user_embeds_mlp = user_embeds_gmf.detach().cpu().numpy(), user_embeds_mlp.detach().cpu().numpy()
                user_sliced_embeds_final_1.append(user_embeds_gmf)
                user_sliced_embeds_final_2.append(user_embeds_mlp)
            for item_embeds in self.sliceItemEmbeddings:
                item_embeds_gmf, item_embeds_mlp = item_embeds(itemIdx)
                item_embeds_gmf, item_embeds_mlp = item_embeds_gmf.detach().cpu().numpy(), item_embeds_mlp.detach().cpu().numpy()
                item_sliced_embeds_final_1.append(item_embeds_gmf)
                item_sliced_embeds_final_2.append(item_embeds_mlp)
                self.user_sliced_embeds_final_1, self.user_sliced_embeds_final_2 = np.stack(user_sliced_embeds_final_1), np.stack(user_sliced_embeds_final_2)
                self.item_sliced_embeds_final_1, self.item_sliced_embeds_final_2 = np.stack(item_sliced_embeds_final_1), np.stack(item_sliced_embeds_final_2)

        else:  # CSMF GraphMF MF
            self.user_sliced_embeds_final = [user_embeds(userIdx).detach().cpu().numpy() for user_embeds in self.sliceUserEmbeddings]
            self.item_sliced_embeds_final = [item_embeds(itemIdx).detach().cpu().numpy() for item_embeds in self.sliceItemEmbeddings]
            self.user_sliced_embeds_final, self.item_sliced_embeds_final = np.stack(self.user_sliced_embeds_final), np.stack(self.item_sliced_embeds_final)

        # if args.interaction == 'NeuCF':
        #     pk.dump(self.user_sliced_embeds_final_1, open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_u1.pkl', 'wb'))
        #     pk.dump(self.user_sliced_embeds_final_2, open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_u2.pkl', 'wb'))
        #     pk.dump(self.item_sliced_embeds_final_1, open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_i1.pkl', 'wb'))
        #     pk.dump(self.item_sliced_embeds_final_2, open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_i2.pkl', 'wb'))
        # else:
        #     pk.dump(self.user_sliced_embeds_final, open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_u.pkl', 'wb'))
        #     pk.dump(self.item_sliced_embeds_final, open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_i.pkl', 'wb'))

        ###################################################################################################################################
        # mean aggregation, not in SISA, RecEraser, just for ours !!!!!!!!!!!! 2023.1.16.18.10
        if args.part_type == 5:
            if self.args.agg2 == 1:
                self.global_interaction = agg_func_param(self, self.global_interaction, args)
        ###################################################################################################################################

        return

    def train_agg_model(self, userIdx, itemIdx):
        userIdx = np.array(userIdx, dtype='int32')
        itemIdx = np.array(itemIdx, dtype='int32')

        if self.args.interaction == 'NeuCF':
            # 取对应的用户的全部桶embedding
            user_embeds1 = t.as_tensor(self.user_sliced_embeds_final_1[:, userIdx]).cuda()
            user_embeds2 = t.as_tensor(self.user_sliced_embeds_final_2[:, userIdx]).cuda()

            item_embeds1 = t.as_tensor(self.item_sliced_embeds_final_1[:, itemIdx]).cuda()
            item_embeds2 = t.as_tensor(self.item_sliced_embeds_final_2[:, itemIdx]).cuda()
            # 2022年12月8日 19:58:01
            user_embeds_1 = self.user_aggregator_1(user_embeds1, userIdx)
            user_embeds_2 = self.user_aggregator_2(user_embeds2, userIdx)

            item_embeds_1 = self.item_aggregator_1(item_embeds1, itemIdx)
            item_embeds_2 = self.item_aggregator_2(item_embeds2, itemIdx)

            user_embeds = user_embeds_1, user_embeds_2
            item_embeds = item_embeds_1, item_embeds_2

        else:  # CSMF GraphMF MF
            user_embeds = t.as_tensor(self.user_sliced_embeds_final[:, userIdx]).cuda()  # 10, 339, 32 -> 10, 10, 32
            item_embeds = t.as_tensor(self.item_sliced_embeds_final[:, itemIdx]).cuda()

            # [10, bs, dim] -> [bs, dim]
            user_embeds = self.user_aggregator(user_embeds, userIdx)
            item_embeds = self.item_aggregator(item_embeds, itemIdx)

        estimated = self.global_interaction(user_embeds, item_embeds)

        return estimated

    ###################################################################################################################################

    # 聚合模型进入测试状态
    def prepare_for_testing(self):
        self.cache = {}

        user_embeds, item_embeds = None, None

        if self.args.interaction == 'NeuCF':
            user_sliced_embeds_final_1 = t.from_numpy(self.user_sliced_embeds_final_1)
            user_sliced_embeds_final_2 = t.from_numpy(self.user_sliced_embeds_final_2)
            item_sliced_embeds_final_1 = t.from_numpy(self.item_sliced_embeds_final_1)
            item_sliced_embeds_final_2 = t.from_numpy(self.item_sliced_embeds_final_2)

            for i in range(self.args.slices):
                user_sliced_embeds_final_1[i] = t.as_tensor(user_sliced_embeds_final_1[i]).cuda()
                user_sliced_embeds_final_2[i] = t.as_tensor(user_sliced_embeds_final_2[i]).cuda()
                item_sliced_embeds_final_1[i] = t.as_tensor(item_sliced_embeds_final_1[i]).cuda()
                item_sliced_embeds_final_2[i] = t.as_tensor(item_sliced_embeds_final_2[i]).cuda()

            user_embeds_1 = self.user_aggregator_1(user_sliced_embeds_final_1.cuda(), t.tensor(range(339)))
            user_embeds_2 = self.user_aggregator_2(user_sliced_embeds_final_2.cuda(), t.tensor(range(339)))

            item_embeds_1 = self.item_aggregator_1(item_sliced_embeds_final_1.cuda(), t.tensor(range(5825)))
            item_embeds_2 = self.item_aggregator_2(item_sliced_embeds_final_2.cuda(), t.tensor(range(5825)))

            self.cache['user1'] = user_embeds_1
            self.cache['user2'] = user_embeds_2
            self.cache['item1'] = item_embeds_1
            self.cache['item2'] = item_embeds_2

        else:  # CSMF GraphMF MF
            user_sliced_embeds_final = t.as_tensor(self.user_sliced_embeds_final)
            item_sliced_embeds_final = t.as_tensor(self.item_sliced_embeds_final)

            for i in range(self.args.slices):
                user_sliced_embeds_final[i] = t.as_tensor(user_sliced_embeds_final[i]).cuda()
                item_sliced_embeds_final[i] = t.as_tensor(item_sliced_embeds_final[i]).cuda()

            user_embeds = self.user_aggregator(user_sliced_embeds_final.cuda(), t.tensor(range(339)))
            item_embeds = self.item_aggregator(item_sliced_embeds_final.cuda(), t.tensor(range(5825)))

        self.cache['user'] = user_embeds
        self.cache['item'] = item_embeds

    # 测试聚合模型
    def test_agg_model(self, user, item):
        if self.args.devices == 'gpu':
            user, item = t.as_tensor(user.detach()).long(), t.as_tensor(item.detach()).long()

        if self.args.interaction == 'NeuCF':
            user_embeds_1 = self.cache['user1'][user]
            user_embeds_2 = self.cache['user2'][user]
            item_embeds_1 = self.cache['item1'][item]
            item_embeds_2 = self.cache['item2'][item]
            user_embeds = user_embeds_1, user_embeds_2
            item_embeds = item_embeds_1, item_embeds_2
        else:  # CSMF GraphMF MF
            user_embeds = self.cache['user'][user]
            item_embeds = self.cache['item'][item]

        estimated = self.global_interaction(user_embeds, item_embeds)

        estimated[estimated < 0] = 0

        return estimated

    ###################################################################################################################################
    # 得到单个切片的模型代码
    def get_slice_model_parameters(self):
        parameters = []

        for params in self.sliceUserEmbeddings[self.sliceId].parameters():
            parameters += [params]

        for params in self.sliceItemEmbeddings[self.sliceId].parameters():
            parameters += [params]

        for params in self.slicedInterFunction[self.sliceId].parameters():
            parameters += [params]

        return parameters

    # 得到聚合模型的全部参数
    def get_agg_model_parameters(self):
        parameters = []
        if self.args.interaction == 'NeuCF':
            for params in self.user_aggregator_1.parameters():
                parameters += [params]
            for params in self.item_aggregator_1.parameters():
                parameters += [params]
            for params in self.user_aggregator_2.parameters():
                parameters += [params]
            for params in self.item_aggregator_2.parameters():
                parameters += [params]
        else:  # CSMF, GraphMF, Pure_MF
            for params in self.user_aggregator.parameters():
                parameters += [params]
            for params in self.item_aggregator.parameters():
                parameters += [params]

        for params in self.global_interaction.parameters():
            parameters += [params]

        return parameters

    def get_attention_model_parameters(self):
        parameters = []
        if self.args.part_type in [2, 5, 6]:
            if self.args.interaction == 'NeuCF':
                for params in self.item_aggregator_1.attention.parameters():
                    parameters += [params]
                for params in self.item_aggregator_2.attention.parameters():
                    parameters += [params]
            else:  # CSMF, GraphMF, Pure_MF
                for params in self.item_aggregator.attention.parameters():
                    parameters += [params]

        elif self.args.part_type == 8:
            if self.args.interaction == 'NeuCF':
                for params in self.user_aggregator_1.attention.parameters():
                    parameters += [params]
                for params in self.user_aggregator_2.attention.parameters():
                    parameters += [params]
            else:  # CSMF, GraphMF, Pure_MF
                for params in self.user_aggregator.attention.parameters():
                    parameters += [params]

        return parameters

    def get_attention_model_parameters2(self):
        parameters = []
        if self.args.part_type in [2, 5, 6]:
            if self.args.interaction == 'NeuCF':
                for params in self.item_aggregator_1.fc.parameters():
                    parameters += [params]
                for params in self.item_aggregator_2.fc.parameters():
                    parameters += [params]
            else:  # CSMF, GraphMF, Pure_MF
                for params in self.item_aggregator.fc.parameters():
                    parameters += [params]
        elif self.args.part_type == 8:
            if self.args.interaction == 'NeuCF':
                for params in self.user_aggregator_1.fc.parameters():
                    parameters += [params]
                for params in self.user_aggregator_2.fc.parameters():
                    parameters += [params]
            else:  # CSMF, GraphMF, Pure_MF
                for params in self.user_aggregator.fc.parameters():
                    parameters += [params]

        return parameters
    ###################################################################################################################################


###################################################################################################################################
def agg_func_param(model, submodel, args):
    param = [[] for _ in range(args.slices)]
    for i in range(args.slices):
        for name, parameters in model.sliceUserEmbeddings[i].named_parameters():
            param[i].append([name, parameters])
        for name, parameters in model.sliceItemEmbeddings[i].named_parameters():
            param[i].append([name, parameters])
        for name, parameters in model.slicedInterFunction[i].named_parameters():
            param[i].append([name, parameters])

    def mean_agg_subfunc_param(model, submodel, string):
        # 获取这个子模型的全部参数
        X = t.tensor([]).cuda()
        for i in range(len(param)):
            for j in range(len(param[i])):
                if param[i][j][0] == string:
                    if len(param[i][j][1].shape) == 1:
                        temp = param[i][j][1].reshape(1, -1)
                    temp = param[i][j][1].unsqueeze(0).cuda()
                    X = t.cat((X, temp))

        X = X.reshape(X.shape[0], X.shape[1], -1)
        Y = t.as_tensor(X, dtype=t.double)
        final_weight = Y.mean(dim=0)

        return final_weight

    # 正式执行区域
    submodel = submodel
    import collections
    final_weight = collections.OrderedDict()

    # 对一个模型的全部子模型聚合
    for name, parameter in submodel.named_parameters():
        temp = mean_agg_subfunc_param(model, submodel, name)

        if temp.shape[1] == 1:
            temp = temp.reshape(-1)

        parameters = t.nn.parameter.Parameter(temp)

        final_weight[name] = parameters

    submodel.load_state_dict(final_weight)

    return submodel
###################################################################################################################################


def quick_train(args, model, first):
    makedir('./pretrain/quick_train')
    return False
    if args.part_type in [6, 8]:
        return False
    try:
        if args.interaction == 'NeuCF':
            model.user_sliced_embeds_final_1 = pk.load(open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_u1.pkl', 'rb'))
            model.user_sliced_embeds_final_2 = pk.load(open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_u2.pkl', 'rb'))
            model.item_sliced_embeds_final_1 = pk.load(open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_i1.pkl', 'rb'))
            model.item_sliced_embeds_final_2 = pk.load(open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_i2.pkl', 'rb'))
        else:  # CSMF GraphMF MF
            model.user_sliced_embeds_final = pk.load(open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_u.pkl', 'rb'))
            model.item_sliced_embeds_final = pk.load(open(f'./pretrain/quick_train/{args.dataset}_interaction{args.interaction}_slices_{args.slices}_density_{args.density:.2f}_part_type{args.part_type}_i.pkl', 'rb'))
        return True
    except:
        if first:
            return False
