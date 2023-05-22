
import h5py

import torch.nn as nn
from time import time
import pandas as pd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from evaluation import eva
import random
import torch
import numpy as np
from sklearn.metrics import davies_bouldin_score,calinski_harabasz_score, silhouette_score
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)

corpora_name ='X' # id of data set
labels ='labels' # labels of dataset
nb_dimension = 768 # embeddings dim


#torch.cuda.set_device(0)

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

        return x_bar, z



class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=256,shuffle=True)  #
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    e_start = time()
    for epoch in range(30):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()

            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            
            #for kmeans please un-comment
          #  kmeans = KMeans(n_clusters = 26, n_init=2, random_state= 2).fit(z.data.cpu().numpy())
            brc = Birch(n_clusters=26).fit(z.data.cpu().numpy())
            labels = brc.predict(z.data.cpu().numpy())
            # Calculate the number of unique labels
            num_clusters = np.unique(labels).shape[0]
            if num_clusters > 1:
                sil = silhouette_score(z.data.cpu().numpy(), labels)
            else:
                sil = 0
            eva(y, labels,sil, epoch )
            e_end = time()
            print("Epoch",{epoch},"took",{e_end-e_start}," sec to run.")
            #eva(y, kmeans.labels_, epoch)
        torch.save(model.state_dict(), corpora_name+'.pkl')
   




model = AE(
        n_enc_1=1000,
        n_dec_1=1000,
        n_input=nb_dimension, #
        n_z=100).cuda()

x = np.loadtxt(corpora_name + '.txt', dtype=float)
print(len(x[0]))
y = np.loadtxt(labels + '.txt', dtype=int)

dataset = LoadDataset(x)
start = time()
pretrain_ae(model, dataset, y)
end = time()
print("Pretraining took ",{end-start}," sec to run.")
