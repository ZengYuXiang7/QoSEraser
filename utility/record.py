#
from utility.utils import *
import pandas as pd
import numpy as np


# 查询是否已做过实验，自适应重做实验
def check_records(args):

    file_address = File_address(args)

    def clear_time(args, df, round):
        # 开始清空
        df = np.array(df)  # 读，转为numpy形式好搞
        start = None
        flag = True
        for i in range(len(df)):  # 获得那个新的一轮未完成的实验的开始位置和结束位置
            if df[i][0] == round and flag:
                start = i
                flag = False
            if df[i][0] == round and not flag:
                end = i
        df = df[:  start, :]  # 只保留前面已经完成的实验
        df = pd.DataFrame(df)  # 转回pandas
        df.columns = ['ROUND', 'SLICES', 'trainging_time']  # 重新命名列名，与原来基本一致
        file = file_address.Final_time_density_csv
        df.to_csv(file, index = False)  # 保存新的结果即为清空


    def clear_txt(args):
        # 这是第一个文件
        file = file_address.Final_result_density_txt

        with open(file, 'r') as f:
            df = f.readlines()
        
        while df[-1] != '\n':  # 如果这一行并不是换行符，就一直删除最后一行，直到遇到换行符，代表本轮清空
            df = df[:-1]

        df = np.array(df)
        np.savetxt(file, X = df, delimiter = '', newline = '', fmt = '%s')  # 重新保存，即为清空

        # 这是第二个文件
        file = file_address.Final_time_density_txt
        with open(file, 'r') as f:
            df = f.readlines()

        while df[-1] != '\n':  # 如果这一行并不是换行符，就一直删除最后一行，直到遇到换行符，代表本轮清空
            df = df[:-1]

        df = np.array(df)
        np.savetxt(file, X = df, delimiter = '', newline = '', fmt = '%s')  # 重新保存，即为清空


    try:
        # 打开第一个文件，如果没有这个文件夹就判断文件夹为空，并且建立
        file = file_address.Final_result_density_csv
        flag = makedir(file_address.result_dir)
        if flag:
            return False, 0  # 所以从头开始（文件夹就判断文件夹为空
        # 如果这个文件夹存在
        df = pd.read_csv(file)

        row = df.shape[0]  # 就看看已经做了几轮实验
        if df.shape[0] == 0:
            return False, 0  # 所以从头开始（文件夹就判断文件夹为空)

        # 同理
        file = file_address.Final_time_density_csv
        flag = makedir(file_address.time_dir)

        if flag:
            return False, 0  # 所以从头开始（文件夹就判断文件夹为空)

        df = pd.read_csv(file)

        if df.shape[0] == 0:  # 如果全部文件夹都在，但是没开始做实验，就从头开始
            return False, 0  # 从头开始


        # 如果上述条件都不满足，就说明已经做了几轮实验了，
        # clear_time(args, df, row + 1)  # 清空新的一轮的结果，row + 1
        # if row - 1 != args.rounds:
        #     clear_txt(args)  # 清空新的一轮结果

        return True, row  # 清空完后就从这一轮继续实验

    except IOError:

        # print('无文件')

        return False, 0  # 一旦这个文件不存在，就从头开始做实验



