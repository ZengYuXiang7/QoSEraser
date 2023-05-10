# coding : utf-8
# Author : yuxiang Zeng
from sklearn.cluster import KMeans
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.nn import *
from tqdm import *
import numpy as np
import pickle as pk
import pandas as pd
import torch as t
from joblib import Parallel, delayed
import numbers

from models.graph_embedding import get_user_embedding


def k_mean(inputs, k, args):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=args.random_state)
    kmeans.fit(inputs)
    return kmeans.labels_


# deep cluster
class AutoEncoder(nn.Module):

    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(args.latent_dim)
        self.dims_list = (args.hidden_dims +
                          args.hidden_dims[:-1][::-1])  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters

        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == self.latent_dim

        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update(
                    {
                        'linear0': nn.Linear(self.input_dim, hidden_dim),
                        'activation0': nn.ReLU()
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            self.hidden_dims[idx-1], hidden_dim),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            self.hidden_dims[idx])
                    }
                )
        self.encoder = nn.Sequential(layers)

        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, self.output_dim),
                    }
                )
            else:
                layers.update(
                    {
                        'linear{}'.format(idx): nn.Linear(
                            hidden_dim, tmp_hidden_dims[idx+1]),
                        'activation{}'.format(idx): nn.ReLU(),
                        'bn{}'.format(idx): nn.BatchNorm1d(
                            tmp_hidden_dims[idx+1])
                    }
                )
        self.decoder = nn.Sequential(layers)

    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
            repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str

    def __str__(self):
        return self.__repr__()

    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output)


class Pretrain_Embeddings(Dataset):
    def __getitem__(self, index):
        return index, self.data[index]

    def __init__(self, data):
        super(Pretrain_Embeddings, self).__init__()

        self.data = t.as_tensor(data).cuda()
        # self.data2 = t.as_tensor(pk.load(open('./user_embeds.pk', 'rb'))).cuda()

    def __len__(self):
        return len(self.data)


class DCN(nn.Module):
    def __init__(self, args):
        super(DCN, self).__init__()
        self.args = args
        self.beta = args.beta  # coefficient of the clustering term
        self.lamda = args.lamda  # coefficient of the reconstruction term
        self.device = t.device('cuda' if args.cuda else 'cpu')

        # Validation check
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))

        if not self.lamda > 0:
            msg = 'lambda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))

        if len(self.args.hidden_dims) == 0:
            raise ValueError('No hidden layer specified.')

        self.kmeans = batch_KMeans(args)
        self.autoencoder = AutoEncoder(args).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = t.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.wd)

    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, X, cluster_id):
        batch_size = X.size()[0]
        rec_X = self.autoencoder(X)
        latent_X = self.autoencoder(X, latent=True)

        # Reconstruction error
        rec_loss = self.lamda * self.criterion(X, rec_X)

        # Regularization term on clustering
        dist_loss = t.tensor(0.).to(self.device)
        clusters = t.FloatTensor(self.kmeans.clusters).to(self.device)
        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = t.matmul(diff_vec.view(1, -1), diff_vec.view(-1, 1))
            dist_loss += 0.5 * self.beta * t.squeeze(sample_dist_loss)

        return rec_loss + dist_loss, rec_loss.detach().cpu().numpy(), dist_loss.detach().cpu().numpy()

    def pretrain(self, train_loader, epoch=100, verbose=True):

        if not self.args.pretrain:
            return

        if not isinstance(epoch, numbers.Integral):
            msg = '`epoch` should be an integer but got value = {}'
            raise ValueError(msg.format(epoch))

        if verbose:
            print('========== Start pretraining ==========')

        rec_loss_list = []

        self.train()
        for e in trange(epoch):
            for batch_idx, (idx, data) in enumerate(train_loader):
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                rec_X = self.autoencoder(data)
                loss = self.criterion(data, rec_X)

                if verbose and batch_idx % self.args.log_interval == 0:
                    msg = 'Epoch: {:02d} | Batch: {:03d} | Rec-Loss: {:.3f}'
                    # print(msg.format(e, batch_idx, loss.detach().cpu().numpy()))
                    rec_loss_list.append(loss.detach().cpu().numpy())

                loss.requires_grad_(True)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.eval()

        # Initialize clusters in self.kmeans after pre-training
        batch_X = []
        for batch_idx, (idx, data) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.autoencoder(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.kmeans.init_cluster(batch_X)

        return rec_loss_list

    def fit(self, epoch, train_loader, verbose=True):

        for batch_idx, (idx, data) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)

            # Get the latent features
            with t.no_grad():
                # print(data.shape)
                latent_X = self.autoencoder(data, latent=True)
                latent_X = latent_X.cpu().numpy()

            # [Step-1] Update the assignment results
            cluster_id = self.kmeans.update_assign(latent_X)

            # [Step-2] Update clusters in bath Kmeans
            elem_count = np.bincount(cluster_id, minlength=self.args.n_clusters)
            for k in range(self.args.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.kmeans.update_cluster(latent_X[cluster_id == k], k)
            # print(self.kmeans.clusters, self.kmeans.clusters.shape)
            # print(cluster_id, cluster_id.shape)
            # [Step-3] Update the network parameters
            loss, rec_loss, dist_loss = self._loss(data, cluster_id)
            loss.requires_grad_(True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose and batch_idx % self.args.log_interval == 0:
                msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.3f} | Rec-' \
                      'Loss: {:.3f} | Dist-Loss: {:.3f}'
                # print(msg.format(epoch, batch_idx, loss.detach().cpu().numpy(), rec_loss, dist_loss))


def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


class batch_KMeans(object):

    def __init__(self, args):
        self.args = args
        self.n_features = args.latent_dim
        self.n_clusters = args.n_clusters
        self.clusters = np.zeros((self.n_clusters, self.n_features))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs

    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)

        return dis_mat

    def init_cluster(self, X, indices=None):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters,
                       n_init=20)
        model.fit(X)
        self.clusters = model.cluster_centers_  # copy clusters

    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] +
                               eta * X[i])
            self.clusters[cluster_idx] = updated_cluster

    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)

        return np.argmin(dis_mat, axis=1)


def solver(args, model, train_loader, test_loader):

    rec_loss_list = model.pretrain(train_loader, args.pre_epoch)
    nmi_list = []
    ari_list = []

    for e in trange(args.epoch):
        model.train()
        model.fit(e, train_loader)
        model.eval()
    print('========== End pretraining ==========')

    return rec_loss_list, nmi_list, ari_list, model


# main function
def deep_cluster(data, args):

    train_loader = t.utils.data.DataLoader(
        Pretrain_Embeddings(data),
        batch_size=339,
        shuffle=False,
        # pin_memory=True,
    )

    # Main body
    model = DCN(args)

    rec_loss_list, nmi_list, ari_list, model = solver(args, model, train_loader, None)

    print('-' * 80)
    for batch in train_loader:
        idx, data = batch
        batch_size = data.size()[0]
        data = data.view(batch_size, -1).to(model.device)

        with t.no_grad():
            latent_X = model.autoencoder(data, latent=True)
            latent_X = latent_X.cpu().numpy()

        label = model.kmeans.update_assign(latent_X)

    # print(label)

    dic = {}
    for item in label:
        if item not in dic:
            dic[item] = 1
        else:
            dic[item] += 1
    # print(dic)

    return label
