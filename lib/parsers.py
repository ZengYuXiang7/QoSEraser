

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()

    # 实验常用
    parser.add_argument('--path', nargs='?', default='./datasets/data/WSDREAM/')
    parser.add_argument('--dataset', type=str, default='rt')
    parser.add_argument('--interaction', type=str, default='NeuCF')

    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--slice_epochs', type=int, default=100)
    parser.add_argument('--slice_lr', type=float, default=1e-3)
    parser.add_argument('--slice_decay', type=float, default=1e-3)
    parser.add_argument('--agg_epochs', type=int, default=10)
    parser.add_argument('--agg_lr', type=float, default=1e-3)
    parser.add_argument('--agg_decay', type=float, default=1e-3)
    parser.add_argument('--att_lr', type=float, default=1e-3)
    parser.add_argument('--att_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--verbose', type=int, default=10)
    parser.add_argument('--devices', type=str, default='gpu')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--slice', type=int, default=1)
    parser.add_argument('--ex', type=str, default='performance')

    # 超参数
    parser.add_argument('--processed', type=int, default=0)  # 数据预处理

    # 切片
    parser.add_argument('--retrain', type=int, default=0)
    parser.add_argument('--slices', type=int, default=5)
    parser.add_argument('--part_type', type=int, default=5)  # 切割方法
    parser.add_argument('--part_iter', type=int, default=20)  # 切割方法

    # NeuCF
    parser.add_argument('--density', type=float, default=0.1)  # 采样率
    parser.add_argument('--dropout', type=float, default=0.1)

    # NeuGraphMF, CSMF
    parser.add_argument('--dimension', type=int, default=32)
    parser.add_argument('--order', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)

    # node2vec
    parser.add_argument('--node2vec', type=int, default=0)
    parser.add_argument('--node2vec_batchsize', type=int, default=32)
    parser.add_argument('--node2vec_walk', type=int, default=15)
    parser.add_argument('--node2vec_dim', type=int, default=128)
    parser.add_argument('--node2vec_epochs', type=int, default=20)
    parser.add_argument('--node2vec_length', type=int, default=8)
    parser.add_argument('--node2vec_windows', type=int, default=3)

    parser.add_argument('--random_state', type=int, default=2022)
    parser.add_argument('--cluster', type=str, default='deep')  # kmeans or deep cluster
    # agg_type
    parser.add_argument('--agg_type', type=str, default='att')

    parser.add_argument('--external_dim', type=int, default=64)

    parser.add_argument('--F', type=int, default=1)
    parser.add_argument('--agg2', type=int, default=0)  # mean fedavg

    parser.add_argument('--agg_function', type=int, default = 1)

    ###################################################################################################################################
    # Deep cluster
    # Dataset parameters
    parser.add_argument('--dir', default='../Dataset/mnist', help='dataset directory')
    parser.add_argument('--input-dim', type=int, default=128, help='input dimension')
    parser.add_argument('--n-classes', type=int, default=10, help='output dimension')
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size for training')
    parser.add_argument('--pre-epoch', type=int, default=1000, help='number of pre-train epochs')
    parser.add_argument('--epoch', type=int, default=1500, help='number of epochs to train')
    parser.add_argument('--pretrain', type=bool, default=True,  help='whether use pre-training')
    # Model parameters
    parser.add_argument('--lamda', type=float, default=1, help='coefficient of the reconstruction loss')
    parser.add_argument('--beta', type=float, default=1, help='coefficient of the regularization term on clustering')
    parser.add_argument('--hidden-dims', default=[64, 64, 100], help='learning rate (default: 1e-4)')
    parser.add_argument('--latent_dim', type=int, default=10, help='latent space dimension')
    parser.add_argument('--n-clusters', type=int, default=10, help='number of clusters in the latent space')
    # Utility parameters
    parser.add_argument('--n-jobs', type=int, default=1, help='number of jobs to run in parallel')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use GPU')
    parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging the training status')
    ###################################################################################################################################

    args = parser.parse_args()

    return args