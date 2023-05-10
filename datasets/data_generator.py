# coding : utf-8
# Author : yuxiang Zeng
import numpy as np


# 一定密度下的采样切分数据集
def get_train_valid_test_dataset(tensor, args):
    quantile = np.percentile(tensor, q=100)
    tensor[tensor > quantile] = 0
    tensor /= quantile
    density = args.density

    mask = np.random.rand(*tensor.shape).astype('float32')  # [0, 1]

    mask[mask > density] = 1
    mask[mask < density] = 0

    train_Tensor = tensor * (1 - mask)

    # size = int(0.05 * np.prod(tensor.shape))
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
