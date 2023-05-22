from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from sklearn.metrics import silhouette_score
import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.cluster import Birch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from evaluation import eva
from collections import Counter



nb_dimension = 768 
class AE(nn.Module):

    def __init__(self, n_enc_1, n_dec_1,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.z_layer = Linear(n_enc_1, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.x_bar_layer = Linear(n_dec_1, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        z = self.z_layer(enc_h1)

        dec_h1 = F.relu(self.dec_1(z))
        x_bar = self.x_bar_layer(dec_h1)

        return x_bar, enc_h1,  z
# set seed
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)

class SDCN(nn.Module):

    def __init__(self, n_enc_1,  n_dec_1, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_dec_1=n_dec_1,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_z)
        self.gnn_3 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, z = self.ae(x)
        
        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1-sigma)*h + sigma*tra1, adj)
        h = self.gnn_3((1-sigma)*h + sigma*z, adj, active=False)
        predict = F.softmax(h, dim=1)

         # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_sdcn(dataset): #only : dataset
    print(len(dataset.x[0]))
    model = SDCN( 1000, 1000,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).to(device)
    print(model)
    print(len(dataset.x[0]))

    optimizer = Adam(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name, args.k)
    adj = adj.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, z = model.ae(data)
        
#   for kmeans initialization    
#    kmeans = KMeans(n_clusters=args.n_clusters, n_init=2, random_state= 2)
#    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
#    y_pred_last = y_pred
#    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

# for initialization   
    brc = Birch(n_clusters=args.n_clusters)
    y_pred = brc.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    n_top_clusters = len(set(y_pred))
    cluster_centers = np.zeros((n_top_clusters, z.shape[1]), dtype=np.float32) 
    np.add.at(cluster_centers, y_pred, z.cpu().float().numpy())  
    cluster_centers /= np.bincount(y_pred, minlength=n_top_clusters)[:, None].astype(np.float32)
    model.cluster_layer.data = torch.tensor(cluster_centers).to(device)
#



    eva(y, y_pred,0, 'pae')
    nb_epochs = 100
    for epoch in range(nb_epochs):
        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            sil= silhouette_score(tmp_q.cpu().numpy(), tmp_q.cpu().numpy().argmax(1))
            eva(y, res1,sil, str(epoch) + 'Q')
            ari, acc = eva(y, res2, sil, str(epoch) + 'Z')
            eva(y, res3,sil, str(epoch) + 'P')
        
        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='X') # 'reut' - 'X'
    parser.add_argument('--k', type=int, default=0) # version(number) of graph generated for this dataset
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=26, type=int)
    parser.add_argument('--n_z', default=100, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name) 
    print(len(dataset.x[0]))

    if args.name == 'X':
        args.n_clusters = 26 #number of class
        args.n_input = nb_dimension


    print(args)
    start = time()
    train_sdcn(dataset)
    end = time()
    print("Training took ",{end-start}," sec to run.")
