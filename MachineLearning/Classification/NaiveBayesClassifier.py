# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 05:19:47 2018

@author: PC Lee

Demonstration of Gaussian naive Bayesian classifier
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

def Gaussian(x, mean, var):
    return 1 / np.sqrt( 2 * np.pi * var) * np.exp( -(x - mean)**2 / (2 * var) )

def Predict(sample, class_prior_hist, classMeanVarTuplelist):
    
    posteriorBelief = []
    for label in range( len( class_prior_hist ) ):
        conditionProbList = []
        for i in range( len( sample) ):
            feature_mean_var = classMeanVarTuplelist[label][i]
            prob = Gaussian( sample[i], feature_mean_var[0], feature_mean_var[1] )
            conditionProbList.append( prob )
        np.log( np.asarray( conditionProbList ) + 1e-6 )
        posteriorBelief.append( np.sum( conditionProbList[label] ) + np.log(class_prior_hist[label] + 1e-6))
    print( posteriorBelief )
    return np.argmax( posteriorBelief )

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
y = y - np.min(y)
X = X.values  #covert to numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)


'''
Compute prior probability distribution
'''

totalSample = X_train.shape[0]
num_feature = X.shape[1]
num_class = np.max(y) + 1
prior_hist = np.zeros( num_class, dtype=int );

for i in range( totalSample ):
    prior_hist[ y_train.values[i] ] += 1 

'''
Compute mean and varianve of each feature for each class
'''

classList = []
for i in range(num_class):
    classList.append([])
for sample, label in zip(X_train, y_train):
    classList[ label ].append( sample )
    
classMeanVarTuplelist = []
for classData in classList:
    mean_var_tuple_list = []
    for feature in range( num_feature):
        data = []
        for item in classData:
            data.append( item[feature] )
        mean = np.mean( data )
        var = np.var( data )
        mean_var_tuple_list.append( (mean, var) )
    classMeanVarTuplelist.append( mean_var_tuple_list )
    
print ( Predict(X_test[0], prior_hist, classMeanVarTuplelist)    )   
            