# coding : utf-8
# Author : yuxiang Zeng
import random
import numpy as np
import pickle as pk
from tqdm import *

from utility.utils import *


# RecEarser
def interaction_based_balanced_parition(tensor, args):
    def E_score2(a, b):
        return np.sum(np.power(a - b, 2))

    try:
        with open(f'./pretrain/RecEarser_{args.slices}.pk', 'rb') as f:
            C = pk.load(f)

    except IOError:
        with open('./pretrain/user_embeds.pk', "rb") as f:
            uidW = pk.load(f)

        with open('./pretrain/item_embeds.pk', "rb") as f:
            iidW = pk.load(f)
        data = []
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                data.append([i, j])

        max_number = 1.2 * (tensor.shape[0] * tensor.shape[1]) // args.slices
        # max_number = 1.2 * Tensor.shape[0] // args.slices

        center_id = random.sample(data, args.slices)

        center_user_value = []
        for i in range(args.slices):
            center_user_value.append([uidW[center_id[i][0]], iidW[center_id[i]][1]])

        C = None
        for iterid in range(args.part_iter):
            C = [{} for _ in range(args.slices)]
            C_number = [0 for _ in range(args.slices)]
            Scores = {}

            for userid in trange(len(data)):
                for sliceid in range(args.slices):
                    score_user = E_score2(np.array(uidW[data[userid][0]]), np.array(center_user_value[sliceid][0]))
                    score_item = E_score2(np.array(iidW[data[userid][1]]), np.array(center_user_value[sliceid][1]))

                    Scores[userid, sliceid] = - score_user * score_item

            Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

            visted = [False for _ in range(len(data))]
            for i in trange(len(Scores)):
                if not visted[Scores[i][0][0]]:
                    if C_number[Scores[i][0][1]] < max_number:
                        if data[Scores[i][0][0]][0] not in C[Scores[i][0][1]]:
                            C[Scores[i][0][1]][data[Scores[i][0][0]][0]] = [
                                data[Scores[i][0][0]][1]]  # 把这个item放入对应的user元祖
                        else:
                            C[Scores[i][0][1]][data[Scores[i][0][0]][0]].append(data[Scores[i][0][0]][1])

                        # print(C[Scores[i][0][1]][data[Scores[i][0][0]][0]])

                        visted[Scores[i][0][0]] = True
                        C_number[Scores[i][0][1]] += 1

            center_user_value_next = []

            for sliceid in trange(args.slices):
                temp_user_value = []
                temp_item_value = []
                user_mean, item_mean = None, None
                for userid in C[sliceid].keys():
                    for itemid in C[sliceid][userid]:
                        temp_user_value.append(uidW[userid])
                        temp_item_value.append(iidW[itemid])
                if len(temp_user_value):
                    user_mean = np.mean(temp_user_value)
                else:
                    user_mean = 0

                if len(temp_item_value):
                    item_mean = np.mean(temp_item_value)
                else:
                    item_mean = 0
                center_user_value_next.append([user_mean, item_mean])

            loss = 0.0

            for sliceid in trange(args.slices):
                score_user = E_score2(np.array(center_user_value_next[sliceid][0]),
                                      np.array(center_user_value[sliceid][0]))
                score_item = E_score2(np.array(center_user_value_next[sliceid][1]),
                                      np.array(center_user_value[sliceid][1]))
                loss += (score_user * score_item)

            center_user_value = center_user_value_next
            log(f'iterid {iterid + 1} : loss = {loss:.30f}')
            for sliceid in range(args.slices):
                log(f'C[{sliceid}] number = {len(list(C[sliceid]))}')

        pk.dump(C, open(f'./pretrain/RecEarser_{args.slices}.pk', 'wb'))

    split_Tensor = []

    row_idx = [[] for _ in range(args.slices)]
    col_idx = [[] for _ in range(args.slices)]

    for sliceid in range(args.slices):
        temp = np.zeros_like(tensor)

        for userid in C[sliceid].keys():
            row_idx[sliceid] += [userid for _ in range(len(C[sliceid][userid]))]
            col_idx[sliceid] += [itemid for itemid in C[sliceid][userid]]

        temp[row_idx[sliceid], col_idx[sliceid]] = tensor[row_idx[sliceid], col_idx[sliceid]]
        split_Tensor.append(temp)

    return split_Tensor