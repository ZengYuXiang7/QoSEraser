# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import *

from models.clusters import k_mean
from utility.utils import *


def balance_part(label, pretrain, args):
    # print('-' * 80)
    C = [[] for _ in range(args.slices)]
    for i in range(args.slices):
        for id, j in enumerate(label):
            if j == i:
                C[i].append(id)
    C = np.array(C, dtype='object')

    def E_score2(value1, value2):
        return np.sum(np.power(value1 - value2, 2))

    center_mean_value = []
    center_ = []
    for i in range(args.slices):
        temp = []
        for j in C[i]:
            temp.append(pretrain[j])
        center_mean_value.append(np.mean(temp, axis=0).tolist())
        center_.append([i, len(C[i]), center_mean_value[i], C[i]])

    center_ = np.array(sorted(center_, key=(lambda x: x[1])), dtype='object')

    vis = [False for _ in range(args.slices)]

    set_changed = True

    max_number = len(label) * 1.2 // args.slices

    # 开始执行
    cnt = 0
    while set_changed:
        set_changed = False

        # 遍历全部类
        for i in range(len(center_)):
            # 如果这个类的数量大于40，就代表他已经不需要合并了，或者他已经被合并过了，跳过
            if center_[i][1] >= max_number or vis[i]:
                continue

            # 寻找距离当前类i 最近的 类goal
            min_dis = 1e9
            goal = -1
            for j in range(i + 1, len(center_)):
                if vis[j]:
                    continue
                # 如果这两个最近的类合并数量大于阈值，跳过
                if center_[i][1] + center_[j][1] <= max_number:
                    # 如果这个类最近，就是他了
                    now_dis_between_i_and_j = E_score2(np.array(center_[i][2]), np.array(center_[j][2]))
                    if now_dis_between_i_and_j < min_dis:
                        min_dis = now_dis_between_i_and_j
                        goal = j
            break

        # 存储合并的类a,和类b
        a = i
        b = goal

        # 获得这两个类的全部信息
        a_ = center_[a]
        b_ = center_[b]

        # 如果确定要合并，goal不是-1
        if goal != -1:
            # print('合并前')
            # for i in center_:
            #     print(i[0], i[1])
            #
            # print(f'{a_[0]} 有 {a_[1]} 个')
            # print(f'{b_[0]} 有 {b_[1]} 个')
            # print(a_[0], b_[0], '合并后该类的数量 : ', a_[1] + b_[1], '个')
            # 让后续加进所有类的时候不重复加，所以已访问
            vis[a_[0]] = True
            vis[b_[0]] = True
            set_changed = True
            cnt += 1

            # 打印合并之前各个类的数量


        # 合并这两个集合
        merge_set = a_[3] + b_[3]
        center_mean_value = []
        # 获得这两个集合中的全部的特征向量并取均值
        for i in range(len(merge_set)):
            center_mean_value.append(pretrain[i])
        mean_value_a_and_b = np.mean(center_mean_value, axis=0).tolist()

        # 这个就是合并的类，标号取两个类的最小，合并数量相加，合并之后，再取这两个类的全部特征取平均作为这个类的均值，以及合并之后集合里面都有哪些用户
        c_ = [min(a_[0], b_[0]), a_[1] + b_[1], mean_value_a_and_b, merge_set]

        # 将合并的集合与剩下的集合整理到一起

        # center_next是更新之后的总样本
        center_next = []
        center_mean_value = []
        for i in range(args.slices):
            temp = []

            # 如果已经合并了，就不需要重复加这两个原来的类
            if vis[i]:
                center_mean_value.append(0)
                continue

            # 就把其他没有合并的类加进来，并且每一个类依然是取均值作为这个类的特征
            for j in C[i]:
                temp.append(pretrain[j])
            center_mean_value.append(np.mean(temp, axis=0).tolist())
            center_next.append([i, len(C[i]), center_mean_value[i], C[i]])

        # 因为我是以最小号 1，9 。取1类为这两个合并之后的标号，所以这个类取消访问
        vis[min(a_[0], b_[0])] = False
        # 并且重置1类中的集合为这个合并的集合
        C[min(a_[0], b_[0])] = merge_set

        # print(c_[0], c_[1], c_[3])

        # 将合并之后的集合加入到新的总样本中
        if goal != -1:
            center_next.append(c_)
            # print('-' * 10)

        # 打印合并之后的全部类的index和数量
        # if goal != -1:
        # print('合并后')
        # for i in center_next:
        #     print(i[0], i[1])

        # 这个排序无关紧要，就是方便理解，按类的数量上下排序
        center_next = np.array(sorted(center_next, key=(lambda x: x[1])), dtype='object')

        # 更新总样本
        center_ = center_next
        # print('-' * 80)

    # print(f'合并了{cnt}次')

    # 将所有超过这个桶的阈值数量的类抽取出来
    extra_set = []
    passed = 0
    for item in center_:
        passed += 1
        if item[1] >= max_number:
            extra_set += item[3]
    np.array(extra_set)

    # 整理信息，获得的是每个用户及其嵌入向量
    now_set = []
    for i in extra_set:
        now_set.append([i, pretrain[i].tolist()])
    now_set = np.array(now_set, dtype='object')
    inputs = np.array(now_set[:, 1].tolist())
    now_set

    # 做第一个想法的kmeans聚类，结果如下所示
    # 除去已经确定的桶，让美国大类再做聚类，放到剩下的十个桶里面 所以是  k = 10 - 已经确定好的桶数量
    label = k_mean(inputs, args.slices - passed, args)

    dic = {}
    for i in range(len(label)):
        if label[i] not in dic:
            dic[label[i]] = 1
        else:
            dic[label[i]] += 1
    dic

    # 和上面一样，整理多出来的用户，美国大类居多
    extra_set = []
    passed = 0
    for item in center_:
        if item[1] >= max_number:
            extra_set += item[3]
        else:
            passed += 1
    np.array(extra_set)

    # 地域二级聚类
    inputs = []
    if args.part_type in [2, 4, 5, 6]:
        ufile = pd.read_csv('./datasets/data/WSDREAM/原始数据/userlist_idx.csv').to_numpy()
        for userid in extra_set:
            inputs.append([ufile[userid][2], ufile[userid][4]])
    elif args.part_type in [7, 8]:
        ufile = pd.read_csv('./datasets/data/WSDREAM/原始数据/wslist_idx.csv').to_numpy()
        for userid in extra_set:
            inputs.append([ufile[userid][2], ufile[userid][4], ufile[userid][6]])

    inputs = np.array(inputs)
    label = k_mean(inputs, args.slices - passed, args)

    dic = {}
    for i in range(len(label)):
        if label[i] not in dic:
            dic[label[i]] = 1
        else:
            dic[label[i]] += 1
    dic

    # 这个时候将各个类放回到对应的集合，方便后续放到桶里面
    extra_set = np.array(extra_set)
    # print(extra_set.shape, label.shape)
    extra_set = np.concatenate([extra_set.reshape(-1, 1), label.reshape(-1, 1)], axis=1)
    extra_set

    # 原来已经确定好的桶是这些
    all_set = {}
    for i in range(passed):
        all_set[i] = center_[i][3]

    # print('原来已经确定的桶')
    # for i in all_set:
    #     print(i, len(all_set[i]))

    # 多出来的用户他们分别对应一些二级类
    extra_set

    # 把这些类加上已经分好的桶，就是对应的新的类
    for i in range(passed, passed + len(dic)):
        temp = []
        for j in range(len(extra_set)):
            if extra_set[j][1] == (i - passed):
                temp.append(extra_set[j][0])
        all_set[i] = temp

    # print('-' * 80)
    log('\tbalance操作结束！')
    # for i in all_set:
    #     print(i, len(all_set[i]))
    # print('-' * 80)

    # pk.dump(all_set, open('./pretrain/user_based_10.pk', 'wb'))

    return all_set