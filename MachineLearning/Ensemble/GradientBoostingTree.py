# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 06:10:55 2018

@author: PC Lee
Demo of gradient boosting tree 
A very nice reference for gradient boosting
http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf

LightGBM
https://github.com/Microsoft/LightGBM/tree/master/examples/python-guide

Catboost
https://github.com/catboost/tutorials

Comparative study of different gradient boosting tree
https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

import lightgbm as lgb
import catboost as cb

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
y = y - np.min(y)
X = X.values  #covert to numpy array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


forest = RandomForestClassifier(criterion='entropy', n_estimators=20, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
print( "score: {}".format( forest.score(X_test, y_test) ) )

gbt = GradientBoostingClassifier( n_estimators=100, learning_rate=0.1, random_state=1)
gbt.fit(X_train, y_train)
print( "score: {}".format( gbt.score(X_test, y_test) ) )

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'num_leaves': 6,
    'metric': ('l1', 'l2'),
    'verbose': 0
}

print('Starting training...')
# train
evals_result = {} 
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_train, lgb_test],
                feature_name=['f' + str(i + 1) for i in range(X_train.shape[-1])],
                categorical_feature=[11],
                evals_result=evals_result,
                verbose_eval=10)

print('Plotting feature importances...')
ax = lgb.plot_importance(gbm, max_num_features=5)
plt.show()

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:{}'.format( mean_squared_error(y_test, y_pred) ** 0.5) )

