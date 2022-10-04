# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:23:41 2022

@author: USER
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

data = pd.read_csv("auto-mpg.csv")
data.drop( 'origin', axis=1, inplace=True )
data.drop( 'car name', axis=1, inplace=True )

hp = data['horsepower'].dropna().median() 
data['horsepower'].fillna(hp, inplace = True)
data = data.astype('int')

x_data = np.array(data[['cylinders','displacement','horsepower','weight','acceleration','model year']])
y_data = np.array(data[['mpg']])

#%%
#High correlation filter
df = pd.DataFrame(x_data)
cor_matrix = df.corr().abs()
print(cor_matrix)

upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
print(upper_tri)

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
print(); print(to_drop)

df1 = df.drop(df.columns[to_drop], axis=1)
print(); print(df1.head())

#%%
#Backward selection
lreg = LinearRegression()

sfs1 = sfs(lreg, k_features=2, forward=False, verbose=1, scoring='neg_mean_squared_error')
sfs1 = sfs1.fit(x_data, y_data)

feat_names = list(sfs1.k_feature_names_)
print(feat_names)

#%%
#PCA
org_data = x_data
target = data.iloc[:,0]

mean = np.mean(org_data, axis= 0)
mean_data = org_data - mean
cov = np.cov(mean_data.T)
cov = np.round(cov, 2)
eig_val, eig_vec = np.linalg.eig(cov)
print("Eigen vectors ", eig_vec)
print("Eigen values ", eig_val, "\n")
indices = np.arange(0,len(eig_val), 1)
indices = ([x for _,x in sorted(zip(eig_val, indices))])[::-1]
eig_val = eig_val[indices]
eig_vec = eig_vec[:,indices]
print("Sorted Eigen vectors ", eig_vec)
print("Sorted Eigen values ", eig_val, "\n")

sum_eig_val = np.sum(eig_val)
explained_variance = eig_val/ sum_eig_val
print(explained_variance)
cumulative_variance = np.cumsum(explained_variance)
print(cumulative_variance)
pca_data = np.dot(mean_data, eig_vec)

#%%
#HERE is for printing the training loss
train = data.sample(frac=0.6, random_state=25)
test = data.drop(train.index)
train.reset_index(inplace=True,drop=True)
test.reset_index(inplace=True,drop=True)

X2 = np.array(train[['cylinders']])
X3 = np.array(train[['displacement']])
X4 = np.array(train[['horsepower']])
X5 = np.array(train[['weight']])
X6 = np.array(train[['acceleration']])
X7 = np.array(train[['model year']])
Y = np.array(train[['mpg']])

x2 = np.array(test[['cylinders']])
x3 = np.array(test[['displacement']])
x4 = np.array(test[['horsepower']])
x5 = np.array(test[['weight']])
x6 = np.array(test[['acceleration']])
x7 = np.array(test[['model year']])
y = np.array(test[['mpg']])

X = np.zeros((len(X2), 2))
Y = Y.reshape(len(Y),1)

X[:,0]=1
X2 = X2.reshape(len(X2))
X3 = X3.reshape(len(X3))
X4 = X4.reshape(len(X4))
X5 = X5.reshape(len(X5))
X6 = X6.reshape(len(X6))
X7 = X7.reshape(len(X7))
X[:,1] = X3
#X[:,2] = X5
#X[:,3] = X4
#X[:,4] = X7
#X[:,5] = X6
#X[:,6] = X5

X_t = X.transpose()
matrix = np.dot(X_t,X)
matrix_inverse = np.linalg.inv(matrix)
para = np.dot(matrix_inverse,np.dot(X_t,Y))

y_pred = np.zeros(len(x2))
y_pred = y_pred + para[0]
y_pred = y_pred + para[1] * x3
#y_pred = y_pred + para[2] * x5
#y_pred = y_pred + para[3] * x4
#y_pred = y_pred + para[4] * x7
#y_pred = y_pred + para[5] * x6
#y_pred = y_pred + para[6] * x5

loss = 0
for i in range(len(x2)):
    loss = loss + (y[i] - y_pred[i])**2
loss = loss/len(x2)
print(loss[0])

y_sum = 0
for i in range(len(y)):
    y_sum+=y[i]
y_average = y_sum/len(y)

y_var = 0
for i in range(len(y)):
    y_var = y_var + (y[i] - y_average)**2
#print(y_var)

R_sq = 1-loss/y_var
print(R_sq[0])

#%%
#HERE is for implementing the elbow method
df = pd.read_csv("Cluster_data.csv")
X = df[['x', 'y']]
x = df[['x']]
y = df[['y']]
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
  
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
  
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_

for key, val in mapping1.items():
    print(f'{key} : {val}')
    
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

for key, val in mapping2.items():
    print(f'{key} : {val}')
    
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

#%%
#HERE is for handcrafting k-means model to plot points
df = pd.read_csv("Cluster_data.csv")
X_train = np.array(df[['x','y']])
true_labels = np.array(df[['y']])
  
def most_common(lst):
    return max(set(lst), key=lst.count)

def euclidean(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

class KMeans:

    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):

        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]

        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx = np.random.choice(range(len(X_train)), size=1, p=dists)[0]  # Indexed @ zero to get val, not array of val
            self.centroids += [X_train[new_centroid_idx]]

        # This method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = self.centroids #init prev_centroids to avoid error
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)

            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1

    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs


# Create a dataset of 2D distributions
centers = 3
X_train = StandardScaler().fit_transform(X_train)

# Fit centroids to dataset
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)

# View results
class_centers, classification = kmeans.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=classification,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         '+',c='r',
         markersize=10,
         )
plt.title("k-means")
plt.show()





