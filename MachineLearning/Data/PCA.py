# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 06:55:00 2018

@author: PCLee

Demonstration of PCA for feature extraction
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

def PlotVarExplanation(eigen_vals):
    plt.figure()
    eigen_sum = np.sum( eigen_vals)
    eigen_ratio = [ i / eigen_sum for i in eigen_vals ]
    eigen_ratio_accum = np.cumsum( eigen_ratio )
    
    plt.bar(range(1, cov_mat.shape[0]+1), eigen_ratio, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1, cov_mat.shape[0]+1), eigen_ratio_accum, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    
df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

'''
calculate covariance matrix of input data 
Note1. the input dimensions for numpy.cov is 'featurex x data size'
Note2. Be caution of the order of the reported eigenvalues from 
       linalg.eig and linalg.eigh
'''

cov_mat = np.cov( X_train_std.T )  
eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
eigen_vals = np.flip( eigen_vals, axis=0) #reverse the order of eigenvalues
eigen_vecs = np.flip( eigen_vecs, axis=0) #reverse the order of eigenvectors

PlotVarExplanation(eigen_vals)

'''
Choose the top-2 explaining factors andconstruct the transformation 
matrix, W.
'''

eigen_pairs = [ (np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range( len(eigen_vals)) ]
W = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]
               ))

'''
Transformation
'''

X_train_pca = np.dot(X_train_std, W) 
X_test_pca = np.dot(X_test_std, W)

'''
Let's see how good it is
'''
lr = LogisticRegression(penalty='l1', C=0.1)
train_accu = []
test_accu = []

for i in range(2, X_train.shape[1], 1):
    W = np.hstack( ( eigen_pairs[k][1][:,np.newaxis] ) for k in range( i)  )
    X_train_pca = np.dot(X_train_std, W) 
    X_test_pca = np.dot(X_test_std, W)
    lr.fit(X_train_pca, y_train)
    train_score = lr.score(X_train_pca, y_train) 
    test_score = lr.score(X_test_pca, y_test) 
    train_accu.append( train_score)
    test_accu.append( test_score)
    print( 'Trainning accuracy: {}'.format(train_score))
    print( 'Test accuracy: {}'.format(test_score))
    
plt.figure()
plt.xlabel('feature number')
plt.ylabel('Accuraccy') 
plt.plot(np.arange(2, 2+len(train_accu)), train_accu, label='Train')
plt.plot(np.arange(2, 2+len(test_accu)), test_accu, label='Test')
plt.legend(loc='best')
plt.show()

lr.fit(X_train_std, y_train)
print('Standard accuracy: {}'.format( lr.score(X_test_std, y_test) ) )
