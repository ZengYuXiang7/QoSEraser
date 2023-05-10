# coding:utf-8
# Author: yuxiang Zeng
from Retrain import EarlyStopping
from lib.data_loader import get_dataloaders
from lib.metrics import ErrMetrics
from datasets.dataset import *
from time import time
from torch.nn import *


# Idea 1 :聚合embedding
###################################################################################################################################
def attention_train(model, train_loader, valid_loader, args):
    training_time = []
    loss_function = t.nn.L1Loss()
    learning_rate = args.agg_lr
    optimizer = t.optim.AdamW(model.global_interaction.parameters(), lr=learning_rate, weight_decay=args.agg_decay)
    optimizer_tf = t.optim.AdamW(model.get_attention_model_parameters(), lr=args.att_lr, weight_decay=args.att_decay)
    optimizer_tf2 = t.optim.AdamW(model.get_attention_model_parameters2(), lr=args.agg_lr, weight_decay=args.agg_decay)

    if args.interaction == 'GraphMF':
        learning_rate = args.agg_lr
        optimizer = t.optim.AdamW(model.global_interaction.parameters(), lr=learning_rate, weight_decay=args.agg_decay)

    if args.devices == 'gpu':
        model = model.cuda()
        loss_function = loss_function.cuda()

    early_stop = EarlyStopping(args, patience=10, verbose=False, delta=0)
    # 正式训练部分
    best_epoch = 0
    validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = 1e5, 1e5, 1e5, 1e5, 1e5
    for epoch in range(args.agg_epochs):
        if args.interaction in ['GraphMF']:
            if epoch % 5 == 0:
                optimizer = t.optim.AdamW(model.global_interaction.parameters(), lr=learning_rate, weight_decay=args.agg_decay)
                optimizer_tf2 = t.optim.AdamW(model.get_attention_model_parameters2(), lr=learning_rate, weight_decay=args.agg_decay)

        t.set_grad_enabled(True)

        t1 = time()
        for trainBatch in train_loader:
            userIdx, itemIdx, mVal = trainBatch
            if args.devices == 'gpu':
                mVal = mVal.cuda()
            pred = model.train_agg_model(userIdx, itemIdx)
            loss = loss_function(pred.to(t.float32), mVal.to(t.float32))
            optimizer.zero_grad()
            optimizer_tf.zero_grad()
            optimizer_tf2.zero_grad()
            loss = loss.requires_grad_()
            loss.backward()
            # t.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
            optimizer.step()
            optimizer_tf2.step()
            optimizer_tf.step()

        t2 = time()
        training_time.append(t2 - t1)

        model.prepare_for_testing()
        validMAE, validRMSE, validNMAE, validMRE, validNPRE = test_agg(model, valid_loader, args)

        if not args.retrain and args.interaction not in ['']:
            early_stop(validMAE, model)
            if early_stop.early_stop:
                break

        if args.record:
            per_epoch_in_txt(args, epoch + 1, validMAE, validRMSE, validNMAE, validMRE, validNPRE, t2 - t1, False)
        log(f'Epoch {(epoch + 1):2d} : loss = {loss:6.6f}  MAE : {validMAE:5.4f}  RMSE : {validRMSE:5.4f}  NMAE : {validNMAE:5.4f}  MRE : {validMRE:5.4f}  NPRE : {validNPRE:5.4f}')

        if validMAE < validBestMAE:
            validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = validMAE, validRMSE, validNMAE, validMRE, validNPRE
            best_epoch = epoch + 1
            t.save(model.state_dict(), f'./pretrain/model_param/{args.dataset}_slices_{args.slices}_round_{args.part_type}_density_{args.density:.2f}_model_parameter.pkl')
        t.set_grad_enabled(True)
    sum_time = sum(training_time[: best_epoch])
    log(f'Best epoch {best_epoch :2d} : MAE : {validBestMAE:5.4f}  RMSE : {validBestRMSE:5.4f}  NMAE : {validBestNMAE:5.4f}  MRE : {validBestMRE:5.4f}  NPRE : {validBestNPRE:5.4f}')
    return validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE, sum_time


# Idea 2 : softmax,
###################################################################################################################################
def baselines_train(model, train_loader, valid_loader, args):
    training_time = []
    loss_function = t.nn.L1Loss()

    learning_rate = args.agg_lr
    optimizer = t.optim.AdamW(model.get_agg_model_parameters(), lr=learning_rate, weight_decay=args.agg_decay)

    if args.interaction == 'GraphMF':
        learning_rate = args.agg_lr
        optimizer = t.optim.AdamW(model.get_agg_model_parameters(), lr=learning_rate, weight_decay=args.agg_decay)

    if args.devices == 'gpu':
        model = model.cuda()
        loss_function = loss_function.cuda()
    early_stop = EarlyStopping(args, patience=10, verbose=False, delta=0)
    # 正式训练部分
    best_epoch = 0
    validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = 1e5, 1e5, 1e5, 1e5, 1e5
    for epoch in range(args.agg_epochs):
        if args.interaction == 'GraphMF':
            if epoch % 5 == 0:
                optimizer = t.optim.AdamW(model.get_agg_model_parameters(), lr=learning_rate, weight_decay=args.agg_decay)

        t.set_grad_enabled(True)
        t1 = time()
        for trainBatch in train_loader:
            userIdx, itemIdx, mVal = trainBatch
            if args.devices == 'gpu':
                mVal = mVal.cuda()
            pred = model.train_agg_model(userIdx, itemIdx)
            loss = loss_function(pred, mVal)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t2 = time()
        training_time.append(t2 - t1)

        model.prepare_for_testing()
        validMAE, validRMSE, validNMAE, validMRE, validNPRE = test_agg(model, valid_loader, args)
        if args.record:
            per_epoch_in_txt(args, epoch + 1, validMAE, validRMSE, validNMAE, validMRE, validNPRE, t2 - t1, False)
        log(f'Epoch {(epoch + 1):2d} : loss = {loss:6.6f}  MAE : {validMAE:5.4f}  RMSE : {validRMSE:5.4f}  NMAE : {validNMAE:5.4f}  MRE : {validMRE:5.4f}  NPRE : {validNPRE:5.4f}')

        if not args.retrain and args.interaction not in ['']:
            early_stop(validMAE, model)
            if early_stop.early_stop:
                break

        if validMAE < validBestMAE:
            validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE = validMAE, validRMSE, validNMAE, validMRE, validNPRE
            best_epoch = epoch + 1
            t.save(model.state_dict(), f'./pretrain/model_param/{args.dataset}_slices_{args.slices}_round_{args.part_type}_density_{args.density:.2f}_model_parameter.pkl')

    sum_time = sum(training_time[: best_epoch])
    log(f'Best epoch {best_epoch :2d} : MAE : {validBestMAE:5.4f}  RMSE : {validBestRMSE:5.4f}  NMAE : {validBestNMAE:5.4f}  MRE : {validBestMRE:5.4f}  NPRE : {validBestNPRE:5.4f}')

    return validBestMAE, validBestRMSE, validBestNMAE, validBestMRE, validBestNPRE, sum_time


def test_agg(model, test_loader, args):
    model.prepare_for_testing()
    writeIdx = 0
    preds = t.zeros((len(test_loader.dataset),)).to('cuda')
    reals = t.zeros((len(test_loader.dataset),)).to('cuda')
    t.set_grad_enabled(False)
    for testBatch in test_loader:
        userIdx, itemIdx, mVal = testBatch
        if args.devices == 'gpu':
            userIdx, itemIdx = userIdx.cuda(), itemIdx.cuda()
        pred = model.test_agg_model(userIdx, itemIdx)
        preds[writeIdx:writeIdx + len(pred)] = pred.cpu()
        reals[writeIdx:writeIdx + len(pred)] = mVal
        writeIdx += len(pred)
    t.set_grad_enabled(True)
    testMAE, testRMSE, testNMAE, testMRE, testNPRE = ErrMetrics(reals.cpu() * model.max_value, preds.cpu() * model.max_value)
    return testMAE, testRMSE, testNMAE, testMRE, testNPRE


def train_aggregate_model(round, model, dataset, args):

    dataset = dataset.full()  # [339, 5825]
    train_loader, valid_loader, test_loader = get_dataloaders(dataset, True, args)
    if args.agg_type in ['att', 'att2']:
        MAE, RMSE, NMAE, MRE, NPRE, training_time = attention_train(model, train_loader, test_loader, args)
    elif args.agg_type in ['softmax', 'mean']:
        MAE, RMSE, NMAE, MRE, NPRE, training_time = baselines_train(model, train_loader, test_loader, args)
    log(f'\t实验 {round:d} 聚合模型训练完毕')

    model.load_state_dict(t.load(f'./pretrain/model_param/{args.dataset}_slices_{args.slices}_round_{args.part_type}_density_{args.density:.2f}_model_parameter.pkl'))
    MAE, RMSE, NMAE, MRE, NPRE = test_agg(model, test_loader, args)
    log(f'Result : MAE : {MAE:5.4f}  RMSE : {RMSE:5.4f}  NMAE : {NMAE:5.4f}  MRE : {MRE:5.4f}  NPRE : {NPRE:5.4f}')

    return MAE, RMSE, NMAE, MRE, NPRE, training_time
###################################################################################################################################

