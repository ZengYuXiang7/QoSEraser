# coding : utf-8
# Author : yuxiang Zeng

from models.transformer import ScaledDotProductAttention, ExternalAttention
from torch.nn import functional as F
from torch.nn import *
from torch import nn
import torch as t
from utility.utils import *

import math


def cat_embedding(embedding, datasets, index, slice_type, args):
    # print(index)
    embedding = embedding.cpu()
    # 3dim -> 2dim
    final_embedding = t.zeros((embedding.shape[1], embedding.shape[2]))
    label = datasets.label
    # 将该桶中的对应的embed加入到final_embedding
    if slice_type == 'user':
        for i in range(len(index)):
            userIdx = index[i]
            this_sliceid = label[userIdx]
            final_embedding[i] = embedding[this_sliceid][i]
    elif slice_type == 'item':
        for i in range(len(index)):
            userIdx = index[i]
            this_sliceid = label[userIdx]
            final_embedding[i] = embedding[this_sliceid][i]
    return final_embedding.cuda()


# 将向量嵌入训练
class UserAggregator(Module):

    def __init__(self, datasets, dim, args):
        super().__init__()
        self.datasets = datasets
        self.args = args
        self.dim = dim

        if args.part_type == 3:
            self.score = Sequential(
                Linear(self.dim, self.dim // 2),
                ReLU(),
                Linear(self.dim // 2, 1)
            )
        elif args.part_type == 8:
            self.attention = ExternalAttention(dim, args.external_dim)
            self.fc = Sequential(
                Linear(self.dim * args.slices, self.dim // 2),
                LayerNorm(self.dim // 2),
                ReLU(),
                Linear(self.dim // 2, self.dim // 2),
                LayerNorm(self.dim // 2),
                ReLU(),
                Linear(self.dim // 2, self.dim),
            )

    def forward(self, user_embeds_list, userIdx):
        # [num_slices, num_users, embed_dim] -> [num_users, embed_dim]
        embeds = t.as_tensor(user_embeds_list)

        if self.args.part_type in [2, 5, 6, 8]:
            agg_embeds = cat_embedding(embeds, self.datasets, userIdx, 'user', self.args)
            return agg_embeds

        # RecEraser
        elif self.args.part_type == 3:
            embeds_w = self.score(embeds.float())
            embeds_w = t.softmax(embeds_w, dim=0)
            agg_embeds = embeds_w * embeds
            return agg_embeds.sum(dim=0)

        # SISA
        elif self.args.part_type == 1:
            agg_embeds = t.mean(embeds, dim=0)
            return agg_embeds


class ItemAggregator(Module):
    def __init__(self, datasets, dim, args):
        super().__init__()
        self.datasets = datasets
        self.args = args
        self.dim = dim

        if args.part_type == 3:
            self.score = Sequential(
                Linear(self.dim, self.dim // 2),
                ReLU(),
                Linear(self.dim // 2, 1)
            )

        elif args.part_type in [2, 5, 6]:
            if args.agg_type == 'att':
                self.attention = ExternalAttention(dim, args.external_dim)
                self.fc = Sequential(
                    Linear(self.dim * args.slices, self.dim // 2),
                    LayerNorm(self.dim // 2),
                    ReLU(),
                    Linear(self.dim // 2, self.dim // 2),
                    LayerNorm(self.dim // 2),
                    ReLU(),
                    Linear(self.dim // 2, self.dim),
                )
                self.norm = t.nn.LayerNorm(dim)
            elif args.agg_type == 'att2':
                self.attention = ScaledDotProductAttention(dim, dim, dim, 1, dropout=0)
                self.fc = Sequential(
                    Linear(self.dim * args.slices, self.dim // 2),
                    LayerNorm(self.dim // 2),
                    ReLU(),
                    Linear(self.dim // 2, self.dim // 2),
                    LayerNorm(self.dim // 2),
                    ReLU(),
                    Linear(self.dim // 2, self.dim),
                )
            elif args.agg_type == 'softmax':
                self.score = Sequential(
                    Linear(self.dim, self.dim // 2),
                    ReLU(),
                    Linear(self.dim // 2, 1)
                )

    def forward(self, item_embeds_list, itemIdx):
        # [num_slices, num_items, embed_dim] -> [num_items, embed_dim]
        embeds = t.as_tensor(item_embeds_list)

        if self.args.part_type == 8:
            agg_embeds = cat_embedding(embeds, self.datasets, itemIdx, 'item', self.args)
            return agg_embeds

        elif self.args.part_type in [2, 5, 6]:
            if self.args.agg_type == 'mean':
                agg_embeds = t.mean(embeds, dim=0)
                return agg_embeds
            elif self.args.agg_type == 'softmax':
                embeds_w = self.score(embeds.float())
                embeds_w = t.softmax(embeds_w, dim=0)
                agg_embeds = embeds_w * embeds
                return agg_embeds.sum(dim=0)
            elif self.args.agg_type == 'att2':
                agg_embeds = self.attention(embeds, embeds, embeds)  # external attention [bs, n, dim] ->
                agg_embeds = agg_embeds.permute(1, 0, 2).reshape(-1, self.dim * self.args.slices)
                agg_embeds = self.fc(agg_embeds)
                return agg_embeds
            elif self.args.agg_type == 'att':
                agg_embeds = self.attention(embeds)  # external attention [bs, n, dim] ->
                agg_embeds = agg_embeds.permute(1, 0, 2).reshape(-1, self.dim * self.args.slices)
                agg_embeds = self.fc(agg_embeds)
                return agg_embeds

        # RecEraser
        elif self.args.part_type == 3:
            embeds_w = self.score(embeds.float())
            embeds_w = t.softmax(embeds_w, dim=0)
            agg_embeds = embeds_w * embeds
            return agg_embeds.sum(dim=0)

        # SISA
        elif self.args.part_type == 1:
            agg_embeds = t.mean(embeds, dim=0)
            return agg_embeds


class FinalMLP(Module):
    def __init__(self, dim):
        super(FinalMLP, self).__init__()
        self.layers = t.nn.Sequential(
            t.nn.Linear(dim * 2, 128),
            t.nn.LayerNorm(128),
            t.nn.ReLU(),
            t.nn.Linear(128, 128),
            t.nn.LayerNorm(128),
            t.nn.ReLU(),
            t.nn.Linear(128, 1)
        )

    def forward(self, userIdx, itemIdx):
        x = t.cat([userIdx, itemIdx], dim = -1)
        x = x.to(t.float32)
        estimated = self.layers(x).sigmoid().flatten()
        return estimated


class ResMLPMixer(Module):

    def __init__(self, dim, depth):
        super(ResMLPMixer, self).__init__()

        self.hori_mix = Linear(dim, dim)

        self.vert_mix = Linear(depth, depth)
        self.linear = Linear(dim, dim)

    def forward(self, x: t.Tensor):
        res = x
        x = self.hori_mix(x)  # type:t.Tensor
        x = F.relu(x, inplace=False)
        x = x + res
        x = self.vert_mix(x.permute(0, 2, 1))
        x = F.relu(x, inplace=False)
        x = x.permute(0, 2, 1)
        x = x + res
        return x


# self.fc = Sequential(
    #     Linear(self.dim * args.slices, self.dim * args.slices // 2),
    #     LayerNorm(self.dim * args.slices // 2),
    #     LeakyReLU(),
    #     Linear(self.dim * args.slices // 2, self.dim * args.slices // 4),
    #     LayerNorm(self.dim * args.slices // 4),
    #     LeakyReLU(),
    #     Linear(self.dim * args.slices // 4, self.dim * args.slices // 8),
    #     LayerNorm(self.dim * args.slices // 8),
    #     LeakyReLU(),
    #     Linear(self.dim * args.slices // 8, self.dim),
    # )