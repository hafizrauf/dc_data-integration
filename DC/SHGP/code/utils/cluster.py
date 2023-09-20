
from sklearn.cluster import KMeans
import numpy as np
from utils.evaluation import eva
import torch
import random
random.seed(555)
np.random.seed(555)
torch.manual_seed(555)
def evaluate_clustering(X, y, n_clusters):


    kmeans = KMeans(n_clusters=n_clusters,random_state=2)
    y_pred = kmeans.fit_predict(X.detach().numpy())

