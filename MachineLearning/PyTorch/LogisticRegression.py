# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 06:37:05 2018

@author: PC Lee
Demo: linear regression using PyTorch 
"""

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

df_wine = pd.read_csv('../Data/winequality-red.csv', sep=';')
df_shape = df_wine.shape
X, y = df_wine.iloc[:, 0:df_shape[1]-1],  df_wine.iloc[:, df_shape[1]-1]
y = y - np.min(y)
X = X.values  #covert to numpy array
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)


num_feature = X.shape[1]
num_class = np.max(y) + 1
model = nn.Linear(num_feature, num_class).to(device)

num_epochs = 500
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(X_train.astype('float32')).to(device)
    targets = torch.from_numpy(y_train.astype('int64')).to(device)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

out = model( torch.from_numpy(X_train.astype('float32')).to(device) )
_, predicted = torch.max(out.data, 1)
print('Train accuracy of the network: {}'.format( np.sum( predicted.cpu().numpy() == y_train) / len(y_train)) )

out = model( torch.from_numpy(X_test.astype('float32')).to(device) )
_, predicted = torch.max(out.data, 1)
print('Test accuracy of the network: {}'.format( np.sum( predicted.cpu().numpy() == y_test) / len(y_test)) )
