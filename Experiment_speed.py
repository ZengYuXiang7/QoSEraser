# coding : utf-8
# Author : yuxiang Zeng
from lib.parsers import get_parser
from trainer.train_speed import main_speed
from utility.utils import *
import torch as t

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
t.set_default_tensor_type(t.FloatTensor)

###################################################################################################################################
###################################################################################################################################
# 主程序
if __name__ == '__main__':
    args = get_parser()
    log('Experiment start!')
    main_speed(args, 0)
    log('Experiment success!\n')
    # fp.close()

###################################################################################################################################
###################################################################################################################################


