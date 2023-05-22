from __future__ import print_function, division
import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import LoadDataset, cluster_acc, ari_sco,revised_rand_index
import tensorflow as tf  
from sklearn.cluster import Birch
import keras.backend as K
import random
import warnings
import pandas as pd
import statistics
from AutoEncoder import AE
from InitializeD import Initialization_D
from Constraint import D_constraint1, D_constraint2
import time
warnings.filterwarnings("ignore")

# set seed
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)
   
class EDESC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_dec_1,
                 n_input,
                 n_z,
                 n_clusters,
                 num_sample,
                 pretrain_path='data/X.pkl'):
        super(EDESC, self).__init__()
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_dec_1=n_dec_1,
            n_input=n_input,
            n_z=n_z)	

        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z, n_clusters))

        
    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # Load pre-trained weights
        self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        print('Load pre-trained model from', path)

    def forward(self, x):
        
        x_bar, z = self.ae(x)
        d = args.d
        s = None
        eta = args.eta
      
        # Calculate subspace affinity
        for i in range(self.n_clusters):	
			
            si = torch.sum(torch.pow(torch.mm(z,self.D[:,i*d:(i+1)*d]),2),1,keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s,si),1)   
        s = (s+eta*d) / ((eta+1)*d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z

    def total_loss(self, x, x_bar, z, pred, target, dim, n_clusters, beta):

	# Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)     
        
        # Subspace clustering loss
        kl_loss = F.kl_div(pred.log(), target)
        
        # Constraints
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)
  
        # Total_loss
        total_loss = reconstr_loss + beta * kl_loss + loss_d1 + loss_d2

        return total_loss

		
def refined_subspace_affinity(s):
    weight = s**2 / s.sum(0)
    return (weight.t() / weight.sum(1)).t()

def pretrain_ae(model):

    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(20):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("Model saved to {}.".format(args.pretrain_path))
    
def train_EDESC():

    model = EDESC(
        n_enc_1=256,
        n_dec_1=256,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        num_sample = args.num_sample,
        pretrain_path=args.pretrain_path).to(device)
    start = time.time()      

    # Load pre-trained model
    model.pretrain('X.pkl')
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)
    x_bar, hidden = model.ae(data)
    #    kmeans = KMeans(n_clusters=args.n_clusters, n_init=2, random_state= 2)
    brc = Birch(n_clusters=args.n_clusters)
     #Get clusters from Consine K-means 
#    X = hidden.data.cpu().numpy()
#    length = np.sqrt((X**2).sum(axis=1))[:,None]
#    X = X / length
#    y_pred = kmeans.fit_predict(X)
 
    # Get clusters from K-means
    y_pred = brc.fit_predict(hidden.data.cpu().numpy())
  
    print("Initial Cluster Centers: ", y_pred)
    
    # Initialize D
    D = Initialization_D(hidden, y_pred, args.n_clusters, args.d)
    D = torch.tensor(D).to(torch.float32)
    accmax = 0
    nmimax = 0
    arimax = 0
    y_pred_last = y_pred
    model.D.data = D.to(device)
    
    model.train()
    e_start = time.time() 
    for epoch in range(20):
        x_bar, s, z = model(data)

        # Update refined subspace affinity
        tmp_s = s.data
        s_tilde = refined_subspace_affinity(tmp_s)

        # Evaluate clustering performance
        y_pred = tmp_s.cpu().detach().numpy().argmax(1)
        np.savetxt('ER_SBERT_EDESC_rep.txt',tmp_s.cpu().detach().numpy(),  fmt= '%.4e')
        sil = silhouette_score(tmp_s.cpu().detach().numpy(), y_pred)                
        delta_label = np.sum(y_pred != y_pred_last).astype(
            np.float32) / y_pred.shape[0]
        y_pred_last = y_pred
        acc = cluster_acc(y, y_pred)
        nmi = nmi_score(y, y_pred)
        ari = ari_sco(y, y_pred)
        n_clus = len(np.unique(y_pred))
        if acc > accmax:
            accmax = acc
        if ari > arimax:
            arimax = ari    
        if nmi > nmimax:
            nmimax = nmi
        if sil > bestsil:
            bestsil = sil               
        print('Iter {}'.format(epoch), ':Current Acc {:.4f}'.format(acc),
                  ':Max Acc {:.4f}'.format(accmax),':Current Ari {:.4f}'.format(ari),
                            ':Max Ari {:.4f}'.format(arimax),', Current nmi {:.4f}'.format(nmi), ':Max nmi {:.4f}'.format(nmimax), ', Number of Clusters {:.4f}'.format(n_clus),' Maxsil {:.4f}'.format(bestsil))
        revised_rand_index(y, y_pred, epoch)
        df = pd.DataFrame(y_pred) 
        count=df.value_counts().tolist()
        print('epoch = '+ str(epoch)+' Median cluster count = '+str(statistics.median(count)))
        print('epoch = '+ str(epoch)+' Mean cluster count = '+str(statistics.mean(count)))
        e_end = time.time() 
        print("Epoch",{epoch},"took",{e_end-e_start}," sec to run.")
        
        ############## Total loss function ######################
        loss = model.total_loss(data, x_bar, z, pred=s, target=s_tilde, dim=args.d, n_clusters = args.n_clusters, beta = args.beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print('Running time: ', end-start)
    return accmax, nmimax, arimax, bestsil
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='EDESC training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=56, type=int)
    parser.add_argument('--d', default=4, type=int)
    parser.add_argument('--n_z', default=104, type=int)
    parser.add_argument('--eta', default=5, type=int)
    #parser.add_argument('--batch_size', default=512, type=int)    
    parser.add_argument('--dataset', type=str, default='dataset1')
    parser.add_argument('--pretrain_path', type=str, default='data/dataset1')
    parser.add_argument('--beta', default=0.1, type=float, help='coefficient of subspace affinity loss')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    args.dataset = 'dataset1'
    if args.dataset == 'dataset1':
        args.pretrain_path = 'data/X.pkl'
        args.n_clusters = 26
        args.n_input = 768
        args.num_sample = 429
        dataset = LoadDataset(args.dataset)   
    print(args)
    bestacc = 0 
    bestnmi = 0
    bestari=0
    bestsil = 0
    for i in range(1):
        acc, nmi, ari, sil = train_EDESC()
        if acc > bestacc:
            bestacc = acc
        if nmi > bestnmi:
            bestnmi = nmi
        if ari > bestari:
            bestari = ari
        if sil > bestsil:
            bestsil = sil        
    print('Best ACC {:.4f}'.format(bestacc), ' Best NMI {:4f}'.format(bestnmi),'Best ARI {:.4f}'.format(bestari),'Best Sil {:.4f}'.format(bestsil))
