# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 05:04:17 2018

@author: PCLee
Gaussian kernel linear discriminant analysis
"""

from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def RBFKernel(xi, xj, gamma):
    return np.exp( -gamma * np.sum((xi - xj)**2) )

def SigmoidKernel(xi, xj, ita, theta):
    return np.tanh(ita * np.dot(xi, xj) + theta)


def Projection(x_new, X, gamma, eigen_vals, eigen_vecs, top_k):
    alphas = np.column_stack( (eigen_vecs[:, i] for i in range(1, top_k)))
    lambdas = [eigen_vals[i] for i in range(1, top_k)]
    pair_dis = np.array( [np.sum(x_new - row)**2 for row in X] )
    K_new = np.exp(-gamma * pair_dis)
    return K_new.dot(alphas / lambdas)
    
df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
y = y - np.min(y)
X = X.values  #covert to numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

'''
Normalize data first, otherwise the exponent will be too large to
cause later numerical computation fail.
'''

normalsc = MinMaxScaler()
X_train_minmax = normalsc.fit_transform( X_train )
X_test_minmax = normalsc.fit_transform( X_test )


'''
Translate features such that mean value is zero
'''

totalSample = X_train.shape[0]
num_feature = X.shape[1]
num_class = np.max(y) + 1

for i in range( num_feature ):
    mean = np.mean( X_train_minmax[:, i] )
    X_train_minmax[:, i] = X_train_minmax[:, i] - mean


'''
Construct kernel matrix
'''

#sq_dists = pdist(X_train_minmax, 'sqeuclidean')
#mat_sq_dists = squareform(sq_dists)

RBF_kernel_var = 1.0
K = np.zeros( (totalSample, totalSample) )
for i in range( totalSample ):
    for j in range (i, totalSample):
        value = RBFKernel(X_train_minmax[i], X_train_minmax[j], RBF_kernel_var)
        K[i][j] = value
        K[j][i] = value
        
eigen_vals, eigen_vecs = np.linalg.eigh( K )
eigen_vals = np.flip( eigen_vals, axis=0) #reverse the order of eigenvalues
eigen_vecs = np.flip( eigen_vecs, axis=0) #reverse the order of eigenvectors



