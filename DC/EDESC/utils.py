from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import h5py
import scipy.io
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
from sklearn import preprocessing
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
from scipy.special import comb

def load_reuters(data_path='./data/dataset1'):
    import os
    # has been shuffled
    x =pd.read_csv('data/dataset1/X.txt', header =None, sep = ' ')
    x = x.to_numpy()
    y =pd.read_csv('data/dataset1/labels.txt', header =None, sep = ' ')
    y = y.to_numpy()
    y=y.flatten()
    
    print(('dataset1 samples', x.shape))
    return x, y
    
def LoadDatasetByName(dataset_name):
    if dataset_name == 'dataset1':
        x, y = load_reuters()
    return x, y

class LoadDataset(Dataset):

    def __init__(self, dataset_name):
        self.x, self.y = LoadDatasetByName(dataset_name)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


#######################################################
# Evaluate Critiron
#######################################################
import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics

def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    # print(l1, l2)
    #为缺失类别进行补充，保障后面使用匈牙利可以一一映射
    if numclass1 > numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                # y_pred.append(i)
                # y_true.append(i)
                y_pred = np.append(y_pred, i)
                y_true = np.append(y_true, i)
                ind += 1

    if numclass1 < numclass2:
        print(l2)
        for i in l2:
            if i in l1:
                pass
            else:
                # y_pred.append(i)
                # y_true.append(i)
                y_pred = np.append(y_pred, i)
                y_true = np.append(y_true, i)
                ind += 1

    l1 = list(set(y_true))
    l2 = list(set(y_pred))
    numclass1 = len(l1)
    numclass2 = len(l2)
    #print(numclass1, numclass2)

    #if numclass1 != numclass2:
     #   print('true = '+str(numclass2))
     #   print('pred = '+str(numclass1))
      #  print('error')
       # return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    y_true = y_true[:len(y_true)-ind]
    new_predict = new_predict[:len(y_pred)-ind]
    y_pred = new_predict

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    # precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    # recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    # f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    # precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    # recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc

def ari_sco(y_true, y_pred):
    ari = ari_score(y_true, y_pred)
    return ari

#define Rand index function
def revised_rand_index(actual, pred, epoch):

    tp_plus_fp = comb(np.bincount(actual), 2).sum()
    tp_plus_fn = comb(np.bincount(pred), 2).sum()
    A = np.c_[(actual, pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(actual))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    print('epoch = '+ str(epoch)+' tp = '+str(tp))
    print('epoch = '+ str(epoch)+' tn = '+str(tn))
    print('epoch = '+ str(epoch)+' fp = '+str(fp))
    print('epoch = '+ str(epoch)+' fn = '+str(fn))
    
    print('epoch = '+ str(epoch)+' Percision is = ' + str(tp/(tp+fp)))
    print('epoch = '+ str(epoch)+' Recall is = ' + str(tp/(tp+fn)))
    p= tp/(tp+fp)
    r = tp/(tp+fn)
    Fmeasure = 2 * ((p*r)/(p+r))
    print('epoch = '+ str(epoch)+ ' updated F1 is = ' + str(Fmeasure))

    return (tp + tn) / (tp + fp + fn + tn)    
