import pandas as pd
import numpy as np
import h5py
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
import random
from time import time
# set seed
random.seed(555)
np.random.seed(555)

topk = 10
corpora_name = 'X' # id of dataset
labels ='labels' # labels of dataset

def construct_graph(features, label, method='heat'):
    fname = 'graph/' + corpora_name + '_graph.txt'
    num = len(label)
    dist = None

    if method == 'heat':
        dist = -0.5 * pair(features) ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk+1))[-(topk+1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    print('error rate: {}'.format(counter / (num * topk)))

# Load the dataset and label that you want to analyse
data = np.loadtxt('data/' + corpora_name + '.txt', dtype=float)
label = np.loadtxt('data/' + labels + '.txt', dtype=int)
start = time()
construct_graph(data, label, 'ncos')
end = time()
print("Graph Generation took ",{end-start}," sec to run.")