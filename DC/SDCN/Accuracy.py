
from sklearn.metrics import rand_score, normalized_mutual_info_score, f1_score,adjusted_rand_score
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import random

 
labels_true = [0, 0, 4, 4, 3, 3]
labels_pred = [1,1,6,6,2, 2]


print('Rand score  is '+ str (adjusted_rand_score(labels_true, labels_pred)))
print('normalized mutual info score  is '+ str (normalized_mutual_info_score(labels_true, labels_pred)))

f1_macro = f1_score(labels_true, labels_pred, average='micro')
print('F-Score is '+ str (f1_macro))



def acc(y_true, y_pred):
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        assert len(y_pred) == len(y_true)
        D = max(max(y_pred), max(y_true)) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(len(y_pred)):
            w[y_pred[i], y_true[i]] += 1
        row, col = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(row, col)]) * 1.0 / len(y_pred)

print('accuracy is '+ str (acc(labels_true, labels_pred)))


