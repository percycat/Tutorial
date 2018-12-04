# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 05:04:17 2018

@author: PCLee
Gaussian kernel linear discriminant analysis
"""

from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def RBFKernel(xi, xj, gamma):
    return np.exp( -gamma * np.sum((xi - xj)**2) )

def SigmoidKernel(xi, xj, ita, theta):
    return np.tanh(ita * np.dot(xi, xj) + theta)

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
y = y - np.min(y)
X = X.values  #covert to numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.995, random_state=0)

'''
Translate features such that mean value is zero
'''
normalsc = MinMaxScaler()
X_train_minmax = normalsc.fit_transform( X_train )
X_test_minmax = normalsc.fit_transform( X_test )

sq_dist = pdist( X_train_minmax, 'sqeuclidean' )
mat_sq_dist = squareform( sq_dist )
K = np.exp( -mat_sq_dist )
N = K.shape[0]
one_n = np.zeros( (N, N ) )
one_n = np.zeros( (N, N ) ) / N
K = K - one_n.dot( K ) - K.dot( one_n ) + one_n.dot( K ).dot( one_n )

totalSample = X_train.shape[0]
num_feature = X.shape[1]
num_class = np.max(y) + 1

for i in range( num_feature ):
    mean = np.mean( X_train_minmax[:, i] )
    X_train_minmax[:, i] = X_train_minmax[:, i] - mean

'''
Construct kernel matrix
'''
RBF_kernel_var = 1.0
K2 = np.zeros( (totalSample, totalSample) )
for i in range( totalSample ):
    for j in range (i, totalSample):
        value = RBFKernel(X_train_minmax[i], X_train_minmax[j], RBF_kernel_var)
        K2[i][j] = value
        K2[j][i] = value



