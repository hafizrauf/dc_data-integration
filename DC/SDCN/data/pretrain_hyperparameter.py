
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
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)

corpora_name ='X' # id of dataset
labels ='labels' # labels of dataset

nb_dimension = 768


#torch.cuda.set_device(0)

class AE(nn.Module):

    def __init__(self, n_enc_1,  n_dec_1,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
       # self.enc_2 = Linear(n_enc_1, n_enc_2)
       # self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_1, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
      #  self.dec_2 = Linear(n_dec_1, n_dec_2)
      #  self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_1, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
       # enc_h2 = F.relu(self.enc_2(enc_h1))
      #  enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h1)

        dec_h1 = F.relu(self.dec_1(z))
     #   dec_h2 = F.relu(self.dec_2(dec_h1))
     #   dec_h3 = F.relu(self.dec_3(dec_h2))
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
    all_acc =[]
    e_start = time()
    for epoch in range(100):
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
            brc = Birch(n_clusters = 26, branching_factor = 20).fit(z.data.cpu().numpy())  #26
            np.savetxt('DNN_rep.csv',z.data.cpu().numpy(),  fmt= '%.4e')
            all_acc.append(eva(y, brc.predict(z.data.cpu().numpy()), epoch))
            e_end = time()
            print("Epoch",{epoch},"took",{e_end-e_start}," sec to run.")
            #eva(y, kmeans.labels_, epoch)
        torch.save(model.state_dict(), corpora_name+'.pkl')
    np.savetxt('AE_pred.csv',brc.predict(z.data.cpu().numpy()),  fmt= '%.4e')    
    np.savetxt('acc_graph.csv',all_acc,  fmt= '%.4e')  




model = AE(
        n_enc_1=1000, #500
        n_dec_1=1000,
        n_input=nb_dimension, #
        n_z=100,).cuda()

x = np.loadtxt(corpora_name + '.txt', dtype=float)
print(len(x[0]))
y = np.loadtxt(labels + '.txt', dtype=int)

dataset = LoadDataset(x)
start = time()
pretrain_ae(model, dataset, y)
end = time()
print("Pretraining took ",{end-start}," sec to run.")
