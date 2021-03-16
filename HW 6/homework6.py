# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:23:04 2021

@author: Mert
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import scipy.spatial as spa
import math

np.random.seed(421)

N1 = 50
N2 = 50
N3 = 50
N4 = 50
N5 = 100

mean1 = np.array([2.5, 2.5])
mean2 = np.array([-2.5, 2.5])
mean3 = np.array([-2.5, -2.5])
mean4 = np.array([+2.5, -2.5])
mean5 = np.array([0, 0])

cov1 = np.array([[0.8, -0.6], [-0.6, 0.8]])
cov2 = np.array([[0.8, 0.6], [0.6, 0.8]])
cov3 = np.array([[0.8, -0.6], [-0.6, 0.8]])
cov4 = np.array([[0.8, 0.6], [0.6, 0.8]])
cov5 = np.array([[1.6, 0], [0, 1.6]])

sample_counter = np.array([N1, N2, N3, N4, N5])
sample_means = np.array([mean1, mean2, mean3, mean4, mean5])
sample_covariances = np.array([cov1, cov2, cov3, cov4, cov5])

data1 = np.random.multivariate_normal(sample_means[0], sample_covariances[0], sample_counter[0])
data2 = np.random.multivariate_normal(sample_means[1], sample_covariances[1], sample_counter[1])
data3 = np.random.multivariate_normal(sample_means[2], sample_covariances[2], sample_counter[2])
data4 = np.random.multivariate_normal(sample_means[3], sample_covariances[3], sample_counter[3])
data5 = np.random.multivariate_normal(sample_means[4], sample_covariances[4], sample_counter[4])

X = np.vstack([data1, data2, data3, data4, data5])

plt.scatter(X[:,0], X[:,1])
plt.axis([-6, 6, -6, 6])
plt.title('Data')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

K = 5

centroids = None
memberships = None
iteration = 1

for i in range(1,3):
    if centroids is None:
        centroids = X[np.random.choice(range(300), K),:]
    else:
        centroids = np.vstack([np.mean(X[memberships == k,], axis = 0) for k in range(K)])
    D = spa.distance_matrix(centroids, X)
    memberships = np.argmin(D, axis = 0)
    
cluster_colors = np.array(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])
cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                           "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
if memberships is None:
    plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
else:
    for c in range(K):
        plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10, 
                 color = cluster_colors[c])
for c in range(K):
    plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
             markerfacecolor = cluster_colors[c], markeredgecolor = "black")
plt.xlabel("x1")
plt.ylabel("x2")

class_means = centroids
class_covariances = [(np.matmul(np.transpose(X[memberships == (c)] - class_means[c]), (X[memberships == (c)] - class_means[c])) / sample_counter[c]) for c in range(K)]
class_priors = [np.mean(memberships == (c)) for c in range(K)]


scores = np.zeros([300,K])



gaus = np.zeros((300,5))
for i in range(100):
    # find new prob matrix
    for a in range(300):
        for b in range(K):
            gaus[a,b]=(1/(2*math.pi*np.linalg.det(class_covariances[b])**0.5))*np.exp(-0.5 * np.transpose( X[a,:]-class_means[b,:]).dot(np.linalg.inv(class_covariances[b])).dot(X[a,:]-class_means[b,:]))

        
    #use probs to find mean cov and prior p
    for k in range(K):
        H = gaus[:,k] * class_priors[k]/np.sum(gaus[i,:].dot(class_priors[k]))
        class_means[k,:]=np.sum(H.dot(X))/np.sum(H)
    
    for k in range(K):
        sumv = 0
        for i in range(300):
            sumv += gaus[i][k]
        class_priors[k] = sumv
        
    
            
    
plt.figure(figsize = (6,6))
for k in range(K):
    plt.plot(X[memberships == k, 0], X[memberships == k, 1], cluster_colors[k], markersize = 10)

















