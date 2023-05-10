# coding : utf-8
# Author : yuxiang Zeng
import platform

from torch.utils.data import DataLoader
from datasets.dataset import ShardedTensorDataset


# 数据集载入，不用动
def get_dataloaders(dataset, agg, args):
    train, valid, test = dataset
    train_set = ShardedTensorDataset(train, False, args)
    valid_set = ShardedTensorDataset(valid, False, args)
    test_set = ShardedTensorDataset(test, False, args)

    if args.retrain == 1:
        agg = True

    if args.retrain:
        bs = args.batch_size
    else:
        bs = args.batch_size * 2

    if agg:
        bs = args.batch_size * 16

    train_loader = DataLoader(
        train_set,
        batch_size= bs,  # 256, 2048
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        num_workers=0 if not agg else 8,
        prefetch_factor=2 if not agg else 4
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=8094,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_set,
        batch_size=8094,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=4
    )

    return train_loader, valid_loader, test_loader
