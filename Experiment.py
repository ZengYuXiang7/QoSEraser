# coding : utf-8
# Author : yuxiang Zeng
import collections
import configparser
import logging
import sys

from lib.parsers import get_parser
from trainer.agg_trainer import *
from trainer.slice_trainer import *

from utility.record import check_records
from models.eraser import TensorEraser, quick_train
from datasets.dataset import *
from utility.utils import *
import numpy as np
import torch as t
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
t.set_default_tensor_type(t.FloatTensor)


# 每轮实验的进行
def run(round, logger, args):
    ###################################################################################################################################
    # 数据模型读取
    df = np.array(load_data(args))
    log('\t原始数据集读取完毕')
    dataset = ShardedTensorDataset(df, True, args)
    set_seed(int(time.time()))
    log('\t分切数据执行完毕')
    model = TensorEraser(339, 5825, dataset, df, args)
    model.max_value = dataset.max_value
    model.label = dataset.label
    log('\t模型加载完毕')
    ###################################################################################################################################
    set_seed(int(time.time()))
    ###################################################################################################################################
    shard_times = []
    # 训练切片模型
    flag2 = quick_train(args, model, True)
    # flag2 = False
    if args.slice and not flag2:
        log('\t准备训练切片模型')
        for sliceId in range(args.slices):
            log(f'\t实验 {round} 模型 {sliceId + 1} 开始分片训练')
            MAE, RMSE, NMAE, MRE, NPRE, train_time = train_sliced_model(round, sliceId, model, dataset, args)
            shard_times.append(train_time)
            log(f'实验 {round:d} 模型 {sliceId + 1} training time = {train_time :.2f} s')
            logger.info(f'实验 {round:d} 模型 {sliceId + 1} MAE = {MAE:.3f}   training time = {train_time :.2f} s')
            log(f'Slice ID {sliceId + 1 : 2d} : MAE = {MAE:.4f}, RMSE = {RMSE:.4f}, NMAE = {NMAE:.4f}, MRE = {MRE:.4f}, NPRE = {NPRE:.4f}')
            if args.record:
                per_slice_time_in_txt(args, round, sliceId + 1, train_time)
                per_slice_time_in_csv(args, round, sliceId + 1, train_time)
        log('\t所有切片模型训练完毕')
        quick_train(args, model, False)
    ##################################################################################################################################

    ###################################################################################################################################
    # 训练聚合模型
    log('\t开始训练聚合模型')
    if not flag2:
        model.prepare_for_aggregation(args)
    MAE, RMSE, NMAE, MRE, NPRE, train_time = train_aggregate_model(round, model, dataset, args)
    log(f'实验 {round:d} : Shards training time = {np.mean(shard_times) :.2f} s')
    log(f'实验 {round:d} : Aggregators training time = {train_time :.2f} s\n')
    logger.info(f'实验 {round:d} : Shards training time = {np.mean(shard_times) :.2f} s')
    logger.info(f'实验 {round:d} : Aggregators training time = {train_time :.2f} s')
    if args.record:
        per_round_agg_time_in_txt(args, round, train_time)
        per_round_agg_time_in_csv(args, round, train_time)
    log('\t聚合模型训练完毕')
    ###################################################################################################################################
    return MAE, RMSE, NMAE, MRE, NPRE, np.mean(shard_times) + train_time


###################################################################################################################################
def main(args, start_round):

    ###################################################################################################################################
    for i, data in pd.read_csv('./utility/hyper_settings.csv').iterrows():
        if data['model'] == args.interaction and data['dataset'] == args.dataset and data['density'] == args.density:
            args.slice_decay = 0.001
            args.agg_decay = 0.001

    config = configparser.ConfigParser()
    config.read('./utility/WSDREAM.conf')

    args = set_settings(args, config)
    ###################################################################################################################################

    ###################################################################################################################################
    # 日志记录
    log(str(args))

    file = './Result/日志/' + args.ex + time.strftime('/%Y-%m-%d %H-%M-%S_', time.localtime(time.time())) + f'{args.dataset}_{args.interaction}_{args.density:.2f}.log'

    makedir(file[:file.find('2023') - 1])
    logging.basicConfig(level=logging.INFO, filename=file, filemode='w')
    logger = logging.getLogger('QoS-Unlearning')

    if args.part_type == 1:
        logger.info(f'SISA 训练框架')
    elif args.part_type == 3:
        logger.info(f'RecEraser 训练框架')
    elif args.part_type == 5:
        logger.info(f'Ours 训练框架')

    logger.info(f'Dataset : {args.dataset}     interaction : {args.interaction}')
    logger.info(f'Density : {(args.density * 100):.2f}%, slice_epochs : {args.slice_epochs}, agg_epochs : {args.agg_epochs}')
    logger.info(f'Part_type : {args.part_type},    slices : {args.slices},       devices : {args.devices}')
    logger.info(f'Slice_lr : {args.slice_lr},    Slice_decay : {args.slice_decay}')
    logger.info(f'Agg_lr : {args.agg_lr},    Agg_decay : {args.agg_decay}')
    logger.info(f'Aggregation function : {args.agg_type}')

    log(f'Dataset : {args.dataset}     interaction : {args.interaction}')
    log(f'Density : {(args.density * 100):.2f}%, slice_epochs : {args.slice_epochs}, agg_epochs : {args.agg_epochs}')
    log(f'Part_type : {args.part_type},    slices : {args.slices},       devices : {args.devices}')
    log(f'Slice_lr : {args.slice_lr},    Slice_decay : {args.slice_decay}')
    log(f'Agg_lr : {args.agg_lr},    Agg_decay : {args.agg_decay}')
    log(f'Aggregation function : {args.agg_type}')
    ###################################################################################################################################

    RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE = [], [], [], [], []
    RunTIME = []
    if start_round != 0:
        MAE, RMSE, NMAE, MRE, NPRE = trained(args)
        for i in range(len(MAE)):
            RunMAE += [MAE[i]]
            RunRMSE += [RMSE[i]]
            RunNMAE += [NMAE[i]]
            RunMRE += [MRE[i]]
            RunNPRE += [NPRE[i]]

    for round in range(args.rounds):
        if round < start_round:
            continue

        log(f'\nRound ' + str(round + 1) + ' experiment start!')
        MAE, RMSE, NMAE, MRE, NPRE, TIME = run(round + 1, logger, args)
        RunMAE += [MAE]
        RunRMSE += [RMSE]
        RunNMAE += [NMAE]
        RunMRE += [MRE]
        RunNPRE += [NPRE]
        RunTIME += [TIME]
        log(f'Round {round + 1} : MAE = {MAE:.3f}, RMSE = {RMSE:.3f}, NMAE = {NMAE:.3f}, MRE = {MRE:.3f}, NPRE = {NPRE:.3f} Train_time = {TIME:.2f} \n')
        logger.info(f'Round {round + 1} : MAE = {MAE:.3f}, RMSE = {RMSE:.3f}, NMAE = {NMAE:.3f}, MRE = {MRE:.3f}, NPRE = {NPRE:.3f} Train_time = {TIME:.2f} \n')

        if args.record:
            per_round_result_in_csv(args, round + 1, MAE, RMSE, NMAE, MRE, NPRE)
            per_round_result_in_txt(args, round + 1, MAE, RMSE, NMAE, MRE, NPRE), print('')

    print('-' * 120)
    log(f'Dataset : {args.dataset}     interaction : {args.interaction}')
    log(f'Density : {(args.density * 100):.2f}%, slice_epochs : {args.slice_epochs}, agg_epochs : {args.agg_epochs}')
    log(f'Part_type : {args.part_type},    slices : {args.slices},       devices : {args.devices}\n')

    if args.rounds != 1:
        for round in range(args.rounds):
            log(f'RoundID {round + 1:} : MAE = {RunMAE[round] :.3f}, RMSE = {RunRMSE[round] :.3f}, NMAE = {RunNMAE[round] :.3f}, MRE = {RunMRE[round] :.3f}, NPRE = {RunNPRE[round] :.3f}')

    log(f'\nDensity {(args.density * 100):.2f}% : MAE = {np.mean(RunMAE, axis=0) :.3f}, RMSE = {np.mean(RunRMSE, axis=0) :.3f}, NMAE = {np.mean(RunNMAE, axis=0) :.3f}, MRE = {np.mean(RunMRE, axis=0) :.3f}, NPRE = {np.mean(RunNPRE, axis=0) :.3f} Train_time = {np.mean(RunTIME):.2f}\n')
    logger.info(f'Density {(args.density * 100):.2f}% : MAE = {np.mean(RunMAE, axis=0) :.3f}, RMSE = {np.mean(RunRMSE, axis=0) :.3f}, NMAE = {np.mean(RunNMAE, axis=0) :.3f}, MRE = {np.mean(RunMRE, axis=0) :.3f}, NPRE = {np.mean(RunNPRE, axis=0) :.3f} Train_time = {np.mean(RunTIME):.2f}')

    if args.record:
        if start_round != args.rounds + 1:
            final_result_in_txt(args, RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE)
            final_result_in_csv(args, RunMAE, RunRMSE, RunNMAE, RunMRE, RunNPRE)

    logging.shutdown()

    # return np.mean(RunMAE, axis = 0)

###################################################################################################################################


###################################################################################################################################
###################################################################################################################################
# 主程序
if __name__ == '__main__':
    # file = './日志/' + time.strftime('%Y-%m-%d/%H-%M-%S_', time.localtime(time.time())) + f'{args.dataset}_{args.interaction}_{args.density:.2f}.log'
    # makedir(file[:16])
    # fp = open(file, "a+")
    # old = sys.stdout
    # sys.stdout = fp  # print重定向到文件

    flag, start_round = False, 0
    args = get_parser()
    if args.record:
        flag, start_round = check_records(args)

    # 自适应中断实验
    if not flag:
        # 是否存储结果
        if args.record and not args.debug:
            per_epoch_result_start(args)
            per_round_result_start(args)
            per_round_result_start_csv(args)
            per_slice_time_start(args)
            per_round_time_start_csv(args)
        log('Experiment start!')
        main(args, start_round)
    else:
        if start_round != args.rounds + 1:
            log('Continue experiment')
        else:
            log('All the experiments have been done!')
        main(args, start_round)
    log('Experiment success!\n')
    # fp.close()

###################################################################################################################################
###################################################################################################################################


