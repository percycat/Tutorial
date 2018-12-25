# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 06:37:05 2018

@author: PC Lee
Demo: linear regression using PyTorch 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
y = y - np.min(y)
X = X.values  #covert to numpy array
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)

'''
Do not forget to pre-process the data, otherwise the optimization process will
not converge
'''
normalsc = MinMaxScaler()
X_train_minmax = normalsc.fit_transform( X_train )
X_test_minmax = normalsc.fit_transform( X_test )

num_feature = X.shape[1]
num_class = np.max(y) + 1
model = nn.Linear(num_feature, num_class)

num_epochs = 200
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(X_train_minmax.astype('float32'))
    targets = torch.from_numpy(y_train.astype('int64'))

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

out = model( torch.from_numpy(X_train_minmax.astype('float32')) )
_, predicted = torch.max(out.data, 1)
print('Train accuracy of the network: {}'.format( np.sum( predicted.numpy() == y_train) / len(y_train)) )

'''
print learned model
'''

print('learned model:')
print('weight: {}'.format(model.weight.view(-1)))
print('bias: {}'.format(model.bias))
