# coding:utf-8
# Author: yuxiang Zeng

from lib.data_loader import get_dataloaders
from lib.metrics import ErrMetrics
from lib.early_stop import EarlyStopping
from datasets.dataset import *
from time import time


###################################################################################################################################
# 训练函数
def train_slice(model, train_loader, valid_loader, args):
    training_time = []
    loss_function = t.nn.L1Loss()
    learning_rate = args.slice_lr
    optimizer_model = t.optim.AdamW(model.get_slice_model_parameters(), lr=learning_rate, weight_decay=args.slice_decay)
    if args.interaction == 'GraphMF':
        lr = args.slice_lr
        optimizer_model = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.slice_decay)
    if args.devices == 'gpu':
        model = model.cuda()
        loss_function = loss_function.cuda()
    early_stop = EarlyStopping(args, patience=20, verbose=False, delta=0)
    max_value = model.max_value

    # 正式训练部分
    best_epoch = 0
    validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = 1e5, 1e5, 1e5, 1e5, 1e5
    for epoch in range(args.slice_epochs):
        if args.interaction == 'GraphMF':
            if epoch % 5 == 0:
                lr /= 2
                optimizer_model = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.slice_decay)
        t.set_grad_enabled(True)

        t1 = time()
        for train_Batch in train_loader:
            userIdx, itemIdx, mVal = train_Batch
            if args.devices == 'gpu':
                userIdx, itemIdx, mVal = userIdx.cuda(), itemIdx.cuda(), mVal.cuda()
            pred = model.train_slice_model(userIdx, itemIdx)
            optimizer_model.zero_grad()
            loss = loss_function(pred, mVal)
            loss.backward()
            optimizer_model.step()
        t2 = time()
        training_time.append(t2 - t1)


        # 设定为评估状态
        t.set_grad_enabled(False)
        model.prepare_test_slice_model()

        validMAE, validRMSE, validNMAE, validMRE, validNPRE = test_slice(model, valid_loader, args)
        if args.verbose and (epoch + 1) % args.verbose == 0:
            log(f'Epoch {(epoch + 1):2d} : MAE : {validMAE:5.4f}  RMSE : {validRMSE:5.4f}  NMAE : {validNMAE:5.4f}  MRE : {validMRE:5.4f}  NPRE : {validNPRE:5.4f}')

        # 切片早停
        if not args.retrain and args.interaction not in ['']:
            early_stop(validMAE, model)
            if early_stop.early_stop:
                break

        if validMAE < validBestMAE:
            validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = validMAE, validRMSE, validNMAE, validMRE, validNPRE
            best_epoch = epoch + 1
            makedir('./pretrain/model_param')
            t.save(model.state_dict(), f'./pretrain/model_param/{args.dataset}_slices_{args.slices}_round_{args.part_type}_density_{args.density:.2f}_model_parameter.pkl')

    sum_time = sum(training_time[: best_epoch])
    log(f'Best epoch {best_epoch :2d} : MAE : {validBestMAE:5.4f}  RMSE : {validBestRMSE:5.4f}  NMAE : {validBestNMAE:5.4f}  MRE : {validBestMRE:5.4f}  NPRE : {validBestNPRE:5.4f}')
    return validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE, sum_time
###################################################################################################################################


###################################################################################################################################
# 测试函数
def test_slice(model, test_loader, args):
    model.prepare_test_slice_model()
    writeIdx = 0
    preds = t.zeros((len(test_loader.dataset),)).to('cuda')
    reals = t.zeros((len(test_loader.dataset),)).to('cuda')
    t.set_grad_enabled(False)
    for testBatch in test_loader:
        userIdx, itemIdx, mVal = testBatch
        if args.devices == 'gpu':
            userIdx, itemIdx = userIdx.cuda(), itemIdx.cuda()
        pred = model.test_slice_model(userIdx, itemIdx)
        preds[writeIdx:writeIdx + len(pred)] = pred.cpu()
        reals[writeIdx:writeIdx + len(pred)] = mVal
        writeIdx += len(pred)
    testMAE, testRMSE, testNMAE, testMRE, testNPRE = ErrMetrics(reals.cpu() * model.max_value, preds.cpu() * model.max_value)
    t.set_grad_enabled(True)
    return testMAE, testRMSE, testNMAE, testMRE, testNPRE
###################################################################################################################################


###################################################################################################################################
# 对该片模型进行训练
def train_sliced_model(round, sliceId, model, datasets, args):

    # 设置切片模型的编号
    model.setSliceId(sliceId)  # [10, 339, 5825]
    # 获得数据载入
    dataset = datasets.get_tensor(sliceId)

    train_loader, valid_loader, test_loader = get_dataloaders(dataset, False, args)
    MAE, RMSE, NMAE, MRE, NPRE, training_time = train_slice(model, train_loader, test_loader, args)

    if args.record:
        per_epoch_in_txt(args, -1, MAE, RMSE, NMAE, MRE, NPRE, training_time, False)
    model.load_state_dict(t.load(f'./pretrain/model_param/{args.dataset}_slices_{args.slices}_round_{args.part_type}_density_{args.density:.2f}_model_parameter.pkl'))
    return MAE, RMSE, NMAE, MRE, NPRE, training_time
###################################################################################################################################
