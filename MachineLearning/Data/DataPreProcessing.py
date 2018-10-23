# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 05:31:15 2018

@author: percy
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-2],  df_wine.iloc[:, 0:df_shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)