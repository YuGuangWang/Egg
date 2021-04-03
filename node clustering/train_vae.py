from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from sklearn.cluster import KMeans
from metrics import clustering_metrics
from mSVD import mSVD
import warnings
import time
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='pubmed', help='type of dataset.')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# GrPool
def grasspool(cur_node_embeddings, p):
    U, S, V = mSVD.apply(cur_node_embeddings)
    print('singular percentage', torch.sum(S[:p]) / torch.sum(S))
    subspace_sym = torch.matmul(U[:, :p], U[:, :p].t())
    return subspace_sym


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, _, _, _, _, true_labels = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])

    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    for rep in range(10):
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        hidden_emb = None
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            recovered, mu, logvar = model(features, adj_norm)
            loss = loss_function(preds=recovered, labels=adj_label,
                                 mu=mu, logvar=logvar, n_nodes=n_nodes,
                                 norm=norm, pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            hidden_emb = mu.data.numpy()

            roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t)
                  )

            if epoch + 1 == args.epochs:
                embed_gr = grasspool(torch.from_numpy(hidden_emb), p=13).numpy()
                kmeans = KMeans(n_clusters=3).fit(embed_gr)
                cluster_ids = kmeans.predict(embed_gr)
                cm = clustering_metrics(true_labels, cluster_ids)
                cm.evaluationClusterModelFromLabel()

        print("Optimization Finished!")

        roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
        print('Test ROC score: ' + str(roc_score))
        print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)