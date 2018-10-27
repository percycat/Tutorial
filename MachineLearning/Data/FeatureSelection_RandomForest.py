# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 05:52:29 2018

@author: PCLee

Demonstration of feature scaling with RandomForest 
for feature selection
The interesting part of RandomForest feauture selection is that
it is robust to the dynamic range of factor value
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import numpy as np

def RandomForestSelection(X, y):
    forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
    forest.fit(X, y)
    importance = forest.feature_importances_
    
    indices = np.argsort(importance)
    
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                labels[indices[f]], 
                                importance[indices[f]]))
    
    plt.bar(range(X.shape[1]), 
            importance[indices],
            color='lightblue', 
            align='center')
    
    plt.xticks(range(X.shape[1]), 
               labels[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.show()

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
labels = df_wine.columns[0: len(df_wine.columns) - 1]
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

stdsc = StandardScaler()
stdsc.mean_ = 0
stdsc.var_ = 1.0
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

normalsc = MinMaxScaler()
X_train_minmax = normalsc.fit_transform( X_train )
X_test_minmax = normalsc.fit_transform( X_test )

plt.figure()
plt.title('Original Feature Importances')
RandomForestSelection(X_train, y_train)

plt.figure()
plt.title('Standard Feature Importances')
RandomForestSelection(X_train_std, y_train)

plt.figure()
plt.title('Normal Feature Importances')
RandomForestSelection(X_train_minmax, y_train)
