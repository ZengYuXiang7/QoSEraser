from time import time

from lib.data_loader import get_dataloaders
from lib.metrics import ErrMetrics
from datasets.dataset import *

# Idea 3 : 选择具体的子模型
###################################################################################################################################
def test_agg3(model, test_loader, args):
    reals, preds = [], []
    # 开始测试
    for testBatch in tqdm(test_loader):
        userIdx, itemIdx, mVal = testBatch
        # if args.devices == 'gpu':
            # userIdx, itemIdx = userIdx.cuda(), itemIdx.cuda()
        pred = model.train_agg_model3(userIdx, itemIdx)
        reals += mVal.tolist()
        preds += pred.tolist()

    reals = np.array(reals)
    preds = np.array(preds)
    testMAE, testRMSE, testNMAE, testMRE, testNPRE = ErrMetrics(reals * model.max_value, preds * model.max_value)

    log(f'Result : MAE : {testMAE:5.4f}  RMSE : {testRMSE:5.4f}  NMAE : {testNMAE:5.4f}  MRE : {testMRE:5.4f}  NPRE : {testNPRE:5.4f}')
    return testMAE, testRMSE, testNMAE, testMRE, testNPRE


def train_aggregate_model3(round, model, dataset, args):

    dataset = dataset.full()

    train_loader, valid_loader, test_loader = get_dataloaders(dataset, False, args)

    MAE, RMSE, NMAE, MRE, NPRE = test_agg3(model, test_loader, args)
    log(f'\t实验 {round:d} 新想法测试完毕!')

    return MAE, RMSE, NMAE, MRE, NPRE
###################################################################################################################################


###################################################################################################################################
###################################################################################################################################


# Idea 4 : Idea 2 和 Idea 3 的结合
###################################################################################################################################
def train_agg4(model, train_loader, valid_loader, args):

    training_time = 0.

    loss_function = t.nn.L1Loss()

    # learning_rate = 8e-3
    # optimizer = t.optim.AdamW(model.get_agg_model_parameters2(), lr=learning_rate, weight_decay=21e-3)
    learning_rate = 1e-3
    optimizer = t.optim.AdamW(model.get_agg_model_parameters(), lr=learning_rate)

    if args.devices == 'gpu':
        model = model.cuda()
        loss_function = loss_function.cuda()

    # 正式训练部分
    best_params = None
    validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = 1e5, 1e5, 1e5, 1e5, 1e5
    validMAE, validRMSE, validNMAE, validMRE, validNPRE = None, None, None, None, None
    for epoch in range(args.agg_epochs):
        t.set_grad_enabled(True)

        t1 = time()
        for trainBatch in train_loader:
            userIdx, itemIdx, mVal = trainBatch
            if args.devices == 'gpu':
                # userIdx, itemIdx = userIdx.cuda(), itemIdx.cuda()
                mVal = mVal.cuda()
            pred = model.train_agg_model4(userIdx, itemIdx)
            loss = loss_function(pred, mVal)
            loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time()

        t.set_grad_enabled(False)

        model.prepare_for_testing()

        reals, preds = [], []
        for validBatch in valid_loader:
            userIdx, itemIdx, mVal = validBatch
            if args.devices == 'gpu':
                # userIdx, itemIdx = userIdx.cuda(), itemIdx.cuda()
                mVal = mVal.cuda()
            pred = model.test_agg_model4(userIdx, itemIdx)
            reals += mVal.tolist()
            preds += pred.tolist()
        reals = np.array(reals)
        preds = np.array(preds)

        # 快速训练
        validMAE, validRMSE, validNMAE, validMRE, validNPRE = ErrMetrics(reals * model.max_value, preds * model.max_value)

        log(f'Epoch {(epoch + 1):2d} : MAE : {validMAE:5.4f}  RMSE : {validRMSE:5.4f}  NMAE : {validNMAE:5.4f}  MRE : {validMRE:5.4f}  NPRE : {validNPRE:5.4f}')

        # 快速训练
        if validMAE < validBestMAE:
            validBestMAE, validBestRMSE, validBestNMAE = validMAE, validRMSE, validNMAE
            training_time += t2 - t1
            best_params = model.state_dict()

        # 继续执行梯度下降
        t.set_grad_enabled(True)

        per_epoch_in_txt(args, epoch + 1, validMAE, validRMSE, validNMAE, validMRE, validNPRE, training_time, False)

    return validMAE, validRMSE, validNMAE, validMRE, validNPRE, best_params, training_time


def test_agg4(model, test_loader, args):
    reals, preds = [], []
    # 开始测试
    for testBatch in tqdm(test_loader):
        userIdx, itemIdx, mVal = testBatch
        # if args.devices == 'gpu':
        #     userIdx, itemIdx = userIdx.cuda(), itemIdx.cuda()
        pred = model.test_agg_model4(userIdx, itemIdx)
        reals += mVal.tolist()
        preds += pred.tolist()

    reals = np.array(reals)
    preds = np.array(preds)
    testMAE, testRMSE, testNMAE, testMRE, testNPRE = ErrMetrics(reals * model.max_value, preds * model.max_value)

    log(f'Result : MAE : {testMAE:5.4f}  RMSE : {testRMSE:5.4f}  NMAE : {testNMAE:5.4f}  MRE : {testMRE:5.4f}  NPRE : {testNPRE:5.4f}')
    return testMAE, testRMSE, testNMAE, testMRE, testNPRE


def train_aggregate_model4(round, model, dataset, args):

    dataset = dataset.full()

    train_loader, valid_loader, test_loader = get_dataloaders(dataset, False, args)

    MAE, RMSE, NMAE, MRE, NPRE, params, training_time = train_agg4(model, train_loader, valid_loader, args)

    model.load_state_dict(params)
    MAE, RMSE, NMAE, MRE, NPRE = test_agg4(model, test_loader, args)
    log(f'\t实验 {round:d} 新想法2测试完毕!')

    return MAE, RMSE, NMAE, MRE, NPRE
###################################################################################################################################

###################################################################################################################################
###################################################################################################################################


# Idea 5 : 结果平均聚合
###################################################################################################################################
def test_agg5(model, test_loader, args):
    reals, preds = [], []
    # 开始测试
    for testBatch in tqdm(test_loader):
        userIdx, itemIdx, mVal = testBatch
        # if args.devices == 'gpu':
        #     userIdx, itemIdx = userIdx.cuda(), itemIdx.cuda()
        pred = model.test_agg_model5(userIdx, itemIdx)
        reals += mVal.tolist()
        preds += pred.tolist()

    reals = np.array(reals)
    preds = np.array(preds)
    testMAE, testRMSE, testNMAE, testMRE, testNPRE = ErrMetrics(reals * model.max_value, preds * model.max_value)

    log(f'Result : MAE : {testMAE:5.4f}  RMSE : {testRMSE:5.4f}  NMAE : {testNMAE:5.4f}  MRE : {testMRE:5.4f}  NPRE : {testNPRE:5.4f}')
    return testMAE, testRMSE, testNMAE, testMRE, testNPRE


def train_aggregate_model5(round, model, dataset, args):

    dataset = dataset.full()

    train_loader, valid_loader, test_loader = get_dataloaders(dataset, False, args)

    MAE, RMSE, NMAE, MRE, NPRE = test_agg5(model, test_loader, args)
    log(f'\t实验 {round:d} 新想法3测试完毕!')

    return MAE, RMSE, NMAE, MRE, NPRE
###################################################################################################################################
