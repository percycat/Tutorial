# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 06:59:54 2018

@author: PCLee
Demonstration of different classification methods
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
'''
z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
# plt.savefig('./figures/sigmoid.png', dpi=300)
plt.show()
'''

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
y = y - np.min(y) + 1
X = X.values  #covert to numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

'''
Linear discriminant analysis (LDA)
'''

'''
compute mean vector of each class
'''
mean_vec = []
for quality in range(1, np.max(y)):
    mean_vec.append( np.mean(X_train_std[y_train == quality], axis = 0) )
    
dim = X.shape[1] #feature dimension

'''
Compute within class covariance matrix
'''
S_w = np.zeros( [dim, dim] )
for quality, mv in zip( range(1, np.max(y)), mean_vec):
    class_scatter = np.cov( X_train_std[ y_train==quality].T )
    S_w += class_scatter

'''
compute between class covariance matrix
'''
mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros( [dim, dim] )
for quality, mv in zip( range(1, np.max(y)), mean_vec):
    n = X[y==1].shape[0]
    mv = mv.reshape(dim, 1)
    mean_overall = mean_overall.reshape(dim, 1)
    S_B += n * (mv - mean_overall).dot( (mv-mean_overall).T )

eigen_vals, eigen_vecs = np.linalg.eigh( np.linalg.inv(S_w).dot(S_B) )
eigen_vals = np.flip( eigen_vals, axis=0) #reverse the order of eigenvalues
eigen_vecs = np.flip( eigen_vecs, axis=0) #reverse the order of eigenvectors


'''
Transformation
'''
top_k =  dim // 2
eigen_pairs = [ (eigen_vals[i], eigen_vecs[:,i]) for i in range( top_k ) ]
W = np.hstack( ( eigen_pairs[k][1][:,np.newaxis] ) for k in range( top_k )  )


'''
Test
'''
lr = LogisticRegression()
X_train_lda = X_train_std.dot(W)
X_test_lda = X_test_std.dot(W)
lr.fit(X_train_lda, y_train)
train_score = lr.score(X_train_lda, y_train) 
test_score = lr.score(X_test_lda, y_test) 
print( 'Trainning accuracy: {}'.format(train_score))
print( 'Test accuracy: {}'.format(test_score))


'''
Using scikit-learn LDA module
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=top_k)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
lr.fit(X_train_lda, y_train)
train_score = lr.score(X_train_lda, y_train) 
test_score = lr.score(X_test_lda, y_test) 
print( 'Scikit-learn Trainning accuracy: {}'.format(train_score))
print( 'Scikit-learn Test accuracy: {}'.format(test_score))