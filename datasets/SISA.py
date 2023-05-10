# coding : utf-8
# Author : yuxiang Zeng
import numpy as np


# SISA
def random_split_to_shard(tensor, args):
    split_Tensor = []

    train_Tensor = tensor

    row_Idx, col_Idx = train_Tensor.nonzero()
    p = np.random.permutation(len(row_Idx))
    row_Idx, col_Idx = row_Idx[p], col_Idx[p]

    for sliceId in range(args.slices):
        startIdx = sliceId * len(row_Idx) // args.slices
        endIdx = startIdx + len(row_Idx) // args.slices

        row_Idx_temp, col_Idx_temp = row_Idx[startIdx: endIdx], col_Idx[startIdx: endIdx]

        temp = np.zeros_like(train_Tensor)

        temp[row_Idx_temp, col_Idx_temp] = train_Tensor[row_Idx_temp, col_Idx_temp]

        split_Tensor.append(temp)  # 1

    return split_Tensor


# SISA2
def random_split_to_shard2(tensor, args):
    split_train_Tensor = []
    split_valid_Tensor = []
    split_test_Tensor = []

    train_Tensor, valid_Tensor, test_Tensor = tensor

    idx = np.arange(339, dtype='int64')
    p = np.random.permutation(len(idx))
    idx = idx[p]

    label = [0 for _ in range(339)]

    for sliceId in range(args.slices):
        startIdx = sliceId * len(idx) // args.slices
        endIdx = startIdx + len(idx) // args.slices

        Idx_temp = idx[startIdx: endIdx]

        for i in range(len(Idx_temp)):
            label[Idx_temp[i]] = sliceId

        temp1 = np.zeros_like(train_Tensor)
        temp2 = np.zeros_like(train_Tensor)
        temp3 = np.zeros_like(train_Tensor)

        temp1[Idx_temp] = train_Tensor[Idx_temp]
        temp2[Idx_temp] = valid_Tensor[Idx_temp]
        temp3[Idx_temp] = test_Tensor[Idx_temp]

        split_train_Tensor.append(temp1)  # 1
        split_valid_Tensor.append(temp2)  # 1
        split_test_Tensor.append(temp3)  # 1

    return split_train_Tensor, split_valid_Tensor, split_test_Tensor, label
