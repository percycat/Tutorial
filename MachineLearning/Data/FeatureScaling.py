# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 05:31:15 2018

@author: PCLee
Demonstration of feature scaling with linear regression model 
for feature selection
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

def FeatureSelection(X, y, sfm):
    supportL = []
    supportL.append(X.shape[1])   
    sfm.fit(X, y)
    n_features = sfm.transform(X).shape[1]
    X_transform = sfm.transform(X)
    while n_features > 2:
        sfm.threshold += 0.1
        X_transform = sfm.transform(X)
        supportL.append( sfm.get_support().sum() )
        n_features = X_transform.shape[1]
    
    plt.figure()
    # Plot the selected two features from X.
    plt.title(
        "Features selected from Wine Quality with "
        "threshold %0.3f." % sfm.threshold)
    feature1 = X_transform[:, 0]
    feature2 = X_transform[:, 1]
    plt.plot(feature1, feature2, 'r.')
    plt.xlabel("Feature number 1")
    plt.ylabel("Feature number 2")
    plt.ylim([np.min(feature2), np.max(feature2)])
    plt.show()
    
    return sfm.get_support(), supportL

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-2],  df_wine.iloc[:, df_shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

stdsc = StandardScaler()
stdsc.mean_ = 0
stdsc.var_ = 2.5
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

normalsc = MinMaxScaler()
X_train_minmax = normalsc.fit_transform( X_train )
X_test_minmax = normalsc.fit_transform( X_test )

lr = LogisticRegression(penalty='l1', C=.1)
sfm = SelectFromModel(lr, threshold=0.25)

original_feature, original_support = FeatureSelection(X_train, y_train, sfm)
print('original: {}'.format(original_feature))
std_feature, std_support= FeatureSelection(X_train_std, y_train, sfm)
print('standard: {}'.format(std_feature))
normal_feature, normal_support= FeatureSelection(X_train_minmax, y_train, sfm)
print('minmax: {}'.format(std_feature))

plt.figure()
plt.plot( original_support, color='blue' )
plt.plot( std_support, color='red' )
plt.plot( normal_support, color='green' )
plt.show()

slot=[]
lr.fit(X_train, y_train)
slot.append( lr.score(X_test, y_test))
print( lr.score(X_test, y_test) )
lr.fit(X_train_std, y_train)
slot.append( lr.score(X_test_std, y_test))
print( lr.score(X_test_std, y_test) )
lr.fit(X_train_minmax, y_train)
slot.append( lr.score(X_test_minmax, y_test))
print( lr.score(X_test_minmax, y_test) )

plt.figure()
plt.bar( ['original', 'standard', 'normal'], slot)