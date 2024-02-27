# : Implement linear regression using Stochastic Gradient Descent via Numpy arrays

#step1 : Load Dataset

import numpy as np
from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()
X = dataset.data
Y = dataset.target[:,np.newaxis] #indexing and reshaping the data

#step 2: Split Dataset

def data_split():
    train_ratio=0.7
    val_ratio=0.2
    n = X.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return X[:n_train, :], Y[:n_train, :],X[n_train:n_train+n_val, :], Y[n_train:n_train+n_val, :], X[n_train+n_val:, :], Y[n_train+n_val:, :]

train_X, train_Y, val_X, val_Y, test_X, test_Y = data_split(X, Y)

#step3:Normalization

def normalize(X, mean, std):
    return (X - mean) / std

mean = np.mean(train_X, axis=0)
std = np.std(train_X, axis=0)

# Save mean and std in  the model
model = {'mean': mean, 'std': std}

train_X = normalize(train_X, mean, std)
val_X = normalize(val_X, mean, std)
test_X = normalize(test_X, mean, std)