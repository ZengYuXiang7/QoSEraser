# coding:utf-8
# Author: yuxiang Zeng
from torch.utils.data import Dataset, DataLoader
from torch.nn import *
from tqdm import *
from time import time
import numpy as np
import pickle as pk
import pandas as pd
import torch as t


# 精度计算
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


if __name__ == '__main__':
    pred = t.Tensor([[1, 2, 3, 4]])
    true = t.Tensor([[2, 1, 4, 5]])
    print(ErrMetrics(pred, true))

