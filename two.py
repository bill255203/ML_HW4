# -*- coding: utf-8 -*-
"""
Created on Fri May 20 02:13:42 2022

@author: USER
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


df = pd.read_csv("Cluster_data.csv")
X = df[['x', 'y']]
x = df[['x']]
y = df[['y']]
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
  
# =============================================================================
# for k in K:
#     # Building and fitting the model
#     kmeanModel = KMeans(n_clusters=k).fit(X)
#     kmeanModel.fit(X)
#   
#     distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
#                                         'euclidean'), axis=1)) / X.shape[0])
#     inertias.append(kmeanModel.inertia_)
#   
#     mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
#                                    'euclidean'), axis=1)) / X.shape[0]
#     mapping2[k] = kmeanModel.inertia_
# 
# for key, val in mapping1.items():
#     print(f'{key} : {val}')
#     
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()
# 
# for key, val in mapping2.items():
#     print(f'{key} : {val}')
#     
# plt.plot(K, inertias, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Inertia')
# plt.title('The Elbow Method using Inertia')
# plt.show()
# =============================================================================

cl_num = 3
data_num = 500
thr = [0.00001, 0.00001, 0.00001]


def dist(x, y, mu_x, mu_y):
    return ((mu_x - x)**2 + (mu_y - y)**2)


def cluster(x, y, mu_x, mu_y):

    cls_ = dict()
    for i in range(data_num):
        dists = []
        for j in range(cl_num):
            distant = dist(x[i], y[i], mu_x[j], mu_y[j])
            dists.append(distant)
        cl = dists.index(min(dists))
        if cl not in cls_:
            cls_[cl] = [(x[i], y[i])]
        elif cl in cls_:
            cls_[cl].append((x[i], y[i]))

    return cls_


def re_mu(cls_, mu_x, mu_y):
    new_muX = []
    new_muY = []

    for key, values in cls_.items():

        if len(values) == 0:
            values.append([mu_x[key], mu_y[key]])

        sum_x = 0
        sum_y = 0
        for v in values:
            sum_x += v[0]
            sum_y += v[1]

        new_mu_x = sum_x / len(values)
        new_mu_y = sum_y / len(values)

        new_muX.append(round(new_mu_x, 2))
        new_muY.append(round(new_mu_y, 2))
    return new_muX, new_muY


def main():

    x = np.random.randint(0, 500, data_num)
    y = np.random.randint(0, 500, data_num)

    mu_x = np.random.randint(0, 500, cl_num)
    mu_y = np.random.randint(0, 500, cl_num)

    cls_ = cluster(x, y, mu_x, mu_y)

    new_muX, new_muY = re_mu(cls_, mu_x, mu_y)

    while any((abs(np.array(new_muX) - np.array(mu_x)) > thr)) != False or any(
        (abs(np.array(new_muY) - np.array(mu_y)) > thr)) != False:
        mu_x = new_muX
        mu_y = new_muY
        cls_ = cluster(x, y, mu_x, mu_y)
        new_muX, new_muY = re_mu(cls_, mu_x, mu_y)

    print('Done')

    plt.scatter(x, y)
    plt.scatter(new_muX, new_muY)
    plt.show()

    colors = ['r', 'b', 'g']
    for key, values in cls_.items():
        cx = []
        cy = []
        for v in values:
            cx.append(v[0])
            cy.append(v[1])
        plt.scatter(cx, cy, color=colors[key])

    plt.show()


if __name__ == '__main__':
    main()