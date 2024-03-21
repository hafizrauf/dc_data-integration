import numpy as np
import scipy.sparse as sp
import torch
import pickle
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.metrics import pairwise_distances as pair


def sp_coo_2_sp_tensor(sp_coo_mat):
    indices = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64))
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def train_val_test_split(label_shape, train_percent):
    rand_idx = np.random.permutation(label_shape)
    val_percent = (1.0 - train_percent) / 2
    idx_train = torch.LongTensor(rand_idx[int(label_shape * 0.0): int(label_shape * train_percent)])
    idx_val = torch.LongTensor(
        rand_idx[int(label_shape * train_percent): int(label_shape * (train_percent + val_percent))])
    idx_test = torch.LongTensor(rand_idx[int(label_shape * (train_percent + val_percent)): int(label_shape * 1.0)])
    return idx_train, idx_val, idx_test


def load_data(dataset,train_percent):

    if dataset=="custom":
        label, ft_dict, adj_dict=load_custom_dataset(train_percent)

    return label, ft_dict, adj_dict


from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import normalize






def load_custom_dataset(train_percent):
     
    data = np.loadtxt('X.txt')
    labels = np.loadtxt('labels.txt').astype(int)

    
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    idx_train, idx_val, idx_test = train_val_test_split(labels.shape[0], train_percent)

    
    data_tensor = torch.FloatTensor(data)
    labels_tensor = torch.LongTensor(labels)

    # Construct graph
    adj_matrix = construct_graph(data, labels)

    
    rows, cols = [], []
    for i, neighbors in adj_matrix.items():
        for j in neighbors:
            rows.append(i)
            cols.append(j)
    values = [1] * len(rows)  
    adj_sparse = coo_matrix((values, (rows, cols)), shape=(len(adj_matrix), len(adj_matrix)))

    adj_dict = {'custom': {'custom': torch.FloatTensor(adj_sparse.toarray()).to_sparse()}}

    ft_dict = {'custom': data_tensor}
    label = {'custom': [labels_tensor, idx_train, idx_val, idx_test]}

    return label, ft_dict, adj_dict




