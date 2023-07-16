import os
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from preprocess import *
from utils import *
import argparse
from sklearn.cluster import KMeans
import pickle
from omegaconf import OmegaConf
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn import metrics
import scipy.io as scio
import time
seed(1)
tf.random.set_seed(1)

from scipy import sparse as sp


# Remove warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from sctpf import SCTPF
from loss import *
from graph_function import *
from numpy.linalg import inv

# Compute cluster centroids, which is the mean of all points in one cluster.
def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default="./ziscDesk-single-cell-data/Wang_Lung", type=str)
    parser.add_argument("--pretrain_epochs", default=400, type=int)
    parser.add_argument("--maxiter", default=300, type=int)
    parser.add_argument("--threshold_2", default=0.95,type=float)
    parser.add_argument("--k_cc", default=2, type=int)
    parser.add_argument("--gpu_option", default="0")
    args = parser.parse_args()
    # Load and preprocess data
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_option
    x, y = prepro( args.dataname + '/data.h5')
    x = np.ceil(x).astype(np.int)
    cluster_number = int(max(y) - min(y) + 1)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y
    adata = normalize(adata, copy=True, highly_genes=500, size_factors=True, normalize_input=True, logtrans_input=True)
    count = adata.X

    # Build model
    adj, adj_n = get_adj(count)
    start_time = time.time()
    model = SCTPF(count, adj=adj, adj_n=adj_n,y=y)


    # Pre-training
    model.pre_train(epochs=args.pretrain_epochs)

    # Compute centers
    Y = model.embedding(count, adj_n)
    from sklearn.cluster import SpectralClustering
    labels = SpectralClustering(n_clusters=cluster_number,affinity="precomputed", assign_labels="discretize",random_state=0).fit_predict(adj)
    centers = computeCentroids(Y, labels)

    # Clustering training
    Cluster_predicted=model.alt_train(y, epochs=args.maxiter, centers=centers,n_update=8,threshold_2=args.threshold_2,k_cc=args.k_cc)

    #Compute evaluation metrics
    if y is not None:
        acc = np.round(cluster_acc(y, Cluster_predicted.y_pred), 5)
        y = list(map(int, y))
        Cluster_predicted.y_pred = np.array(Cluster_predicted.y_pred)
        nmi = np.round(metrics.normalized_mutual_info_score(y, Cluster_predicted.y_pred), 5)
        ari = np.round(metrics.adjusted_rand_score(y, Cluster_predicted.y_pred), 5)
        print('ACC= %.4f, NMI= %.4f, ARI= %.4f'
            % (acc, nmi, ari))
    end_time = time.time()
    print('execution time: ', end_time-start_time, 'sec')

