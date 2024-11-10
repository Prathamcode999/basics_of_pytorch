import torch
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np


n_pts = 100
centers = [[-0.5,0.5],[0.5,-0.5]]
X,y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4) # make_blobs is dataset used for classification

# dont confuse the X,y for line equation, 
# X is the corrdinates of datapoint in 2 array whereas y stores the clusters example:
# Let’s say you have n_samples=5 and centers=2 (meaning two clusters, 0 and 1). The output might look something like this:
# X = [[1.5, 2.0], [1.0, 1.5], [7.0, 8.5], [6.5, 7.8], [6.7, 8.1]]
# y = [0, 0, 1, 1, 1]
# X: These are the coordinates of the 5 data points in two-dimensional space.
# y: These are the cluster labels for each point. Points [1.5, 2.0] and [1.0, 1.5] belong to cluster 0, while the other three points belong to cluster 1. 

plt.scatter(X[y==0,0], X[y==0, 1]) #0 index data
plt.scatter(X[y==1,0], X[y==1, 1]) #1 index data, you can see by printing print(X)
plt.show()

x_data = torch.tensor(X)
y_data = torch.tensor(y) # we have to convert the data to tensor to train the data


