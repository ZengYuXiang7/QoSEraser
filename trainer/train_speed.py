# coding:utf-8
# Author: yuxiang Zeng
import collections
import configparser

from lib.data_loader import get_dataloaders
from lib.metrics import ErrMetrics
from lib.early_stop import EarlyStopping
from datasets.dataset import *
from time import time
from models.eraser import TensorEraser


# Idea 1 :聚合embedding
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

    sum_time = sum(training_time)
    return sum_time
###################################################################################################################################


###################################################################################################################################
# 对该片模型进行训练
def slice_speed(round, sliceId, model, datasets, args):
    model.setSliceId(sliceId)  # [10, 339, 5825]
    dataset = datasets.get_tensor(sliceId)
    train_loader, valid_loader, test_loader = get_dataloaders(dataset, False, args)
    training_time = train_slice(model, train_loader, test_loader, args)
    return training_time
###################################################################################################################################

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

    for epoch in trange(args.agg_epochs):
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
            optimizer.step()
            optimizer_tf2.step()
            optimizer_tf.step()
        t2 = time()
        training_time.append(t2 - t1)
        t.set_grad_enabled(True)
    sum_time = sum(training_time)
    return sum_time

def agg_speed(round, model, dataset, args):

    dataset = dataset.full()  # [339, 5825]
    train_loader, valid_loader, test_loader = get_dataloaders(dataset, True, args)
    training_time = attention_train(model, train_loader, test_loader, args)

    return training_time
###################################################################################################################################

def record_speed(args):
    ###################################################################################################################################
    # 数据模型读取
    df = np.array(load_data(args))
    dataset = ShardedTensorDataset(df, True, args)
    model = TensorEraser(339, 5825, dataset, df, args)
    model.label = dataset.label
    # print(args.slices, args.slice_epochs, args.agg_epochs)
    ###################################################################################################################################
    ###################################################################################################################################
    shard_times = []
    # 训练切片模型
    for sliceId in trange(args.slices):
        train_time = slice_speed(round, sliceId, model, dataset, args)
        shard_times.append(train_time)
    ##################################################################################################################################

    ###################################################################################################################################
    # 训练聚合模型
    model.prepare_for_aggregation(args)
    train_time = agg_speed(round, model, dataset, args)
    ###################################################################################################################################
    if args.slices == 5:
        print(np.mean(shard_times), train_time)
    return np.mean(shard_times) + train_time


###################################################################################################################################
# 执行程序
def main_speed(args, start_round):
    log(str(args))
    ###################################################################################################################################
    config = configparser.ConfigParser()
    config.read('./utility/WSDREAM.conf')
    args = set_settings(args, config)
    ###################################################################################################################################
    results = collections.defaultdict(list)
    speed = {
        'rt' : {
            5: {
                'slice_epochs': {'NeuCF': 15, 'CSMF': 10, 'MF': 20, 'GraphMF': 30},
                'agg_epochs': {'NeuCF': 15, 'CSMF': 15, 'MF': 15, 'GraphMF': 35}
            },
            10: {
                'slice_epochs': {'NeuCF': 15, 'CSMF': 10, 'MF': 20, 'GraphMF': 30},
                'agg_epochs': {'NeuCF': 18, 'CSMF': 20, 'MF': 20, 'GraphMF': 35}
            },
            15: {
                'slice_epochs': {'NeuCF': 15, 'CSMF': 10, 'MF': 20, 'GraphMF': 30},
                'agg_epochs': {'NeuCF': 21, 'CSMF': 25, 'MF': 25, 'GraphMF': 45}
            },
            20: {
                'slice_epochs': {'NeuCF': 15, 'CSMF': 10, 'MF': 20, 'GraphMF': 30},
                'agg_epochs': {'NeuCF': 24, 'CSMF': 30, 'MF': 30, 'GraphMF': 50}
            }
        },
        'tp': {
            5: {
                'slice_epochs': {'NeuCF': 20, 'CSMF': 15, 'MF': 25, 'GraphMF': 40},
                'agg_epochs': {'NeuCF': 18, 'CSMF': 20, 'MF': 20, 'GraphMF': 40}
            },
            10: {
                'slice_epochs': {'NeuCF': 20, 'CSMF': 15, 'MF': 25, 'GraphMF': 40},
                'agg_epochs': {'NeuCF': 21, 'CSMF': 25, 'MF': 25, 'GraphMF': 35}
            },
            15: {
                'slice_epochs': {'NeuCF': 20, 'CSMF': 15, 'MF': 25, 'GraphMF': 40},
                'agg_epochs': {'NeuCF': 24, 'CSMF': 30, 'MF': 30, 'GraphMF': 50}
            },
            20: {
                'slice_epochs': {'NeuCF': 20, 'CSMF': 15, 'MF': 25, 'GraphMF': 40},
                'agg_epochs': {'NeuCF': 28, 'CSMF': 35, 'MF': 35, 'GraphMF': 55}
            }
        }
    }
    args.verbose = 0
    # for slices in [10, 15, 20]:
    #     for dataset in ['rt', 'tp']:
    #         for inter in ['NeuCF', 'CSMF', 'MF', 'GraphMF']:
    #             args.interaction = inter
    #             for round in range(args.rounds):
    #                 args.slices = slices
    #                 args.slice_epochs = speed[dataset][slices]['slice_epochs'][inter]
    #                 args.agg_epochs = speed[dataset][slices]['agg_epochs'][inter]
    #                 elapsed = record_speed(args)
    #                 results[str(slices) + dataset + inter].append(elapsed)
    #                 # print(results[str(slices) + dataset + inter])
    #         for inter in ['NeuCF', 'CSMF', 'MF', 'GraphMF']:
    #             print(f'{dataset}-----shard {slices} inference Time: {inter}-{np.mean(results[str(slices) + dataset + inter]):.2f}s')

    for dataset in ['rt', 'tp']:
        for slices in [10]:
            for inter in ['GraphMF']:
                args.interaction = inter
                for round in range(args.rounds):
                    args.slices = slices
                    args.slice_epochs = speed[dataset][slices]['slice_epochs'][inter]
                    args.agg_epochs = speed[dataset][slices]['agg_epochs'][inter]
                    elapsed = record_speed(args)
                    results[str(slices) + dataset + inter].append(elapsed)
                    # print(results[str(slices) + dataset + inter])
            for inter in ['NeuCF', 'CSMF', 'MF', 'GraphMF']:
                print(f'{dataset}-----shard {slices} inference Time: {inter}-{np.mean(results[str(slices) + dataset + inter]):.2f}s')
    print(results)

###################################################################################################################################
