# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 06:57:30 2018

@author: PC Lee
Demo of scikit-learn random forest 
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
y = y - np.min(y)
X = X.values  #covert to numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)
print( "score: {}".format( tree.score(X_test, y_test) ) )

forest = RandomForestClassifier(criterion='entropy', n_estimators=20, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
print( "score: {}".format( forest.score(X_test, y_test) ) )

AdaBosst = AdaBoostClassifier(base_estimator=forest, n_estimators=50, learning_rate=0.1, random_state=0)
AdaBosst.fit(X_train, y_train)
print( "score: {}".format( AdaBosst.score(X_test, y_test) ) )