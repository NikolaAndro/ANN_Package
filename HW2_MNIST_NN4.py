# %% [markdown]
# ### 1. Import Libraries

# %%
import os
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# %matplotlib inline
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# %% [markdown]
# ### 2. Import from mlcblab

# %%
from mlcvlab.models.nn4 import NN4
from mlcvlab.nn.losses import l2
from mlcvlab.optim.sgd import SGD
from mlcvlab.optim.async_sgd import async_sgd
from mlcvlab.optim.sync_sgd import sync_sgd

# %% [markdown]
# ### 3. Set Seed

# %%
np.random.seed(42)

# %% [markdown]
# ### 4. Helper functions

# %%
def load_dataset():
    '''Loads the whole dataset with true labels included.'''
    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    return x,y

def prepare_data(x, y):
    '''Converts 10-ary labels in binary labels. If even then label is 1 otherwise 0.'''
    y = y.astype(int)
    y = y.reshape(len(y),1)
    y =  (y+1) % 2
    return x, y

def split_train_test(x,y):
    '''Partitioning the dataset into 10,000 test samles and the remaining 60,000 as training samples. 
    The shape  of the data will be M x N where M = 784 and N = 60000 for X and N = 10000 for y.'''   
    X_train, X_test = x[:60000].T, x[60000:].T
    y_train, y_test = y[:60000].T, y[60000:].T
    
    # adding -1 to the end of every x  as a bias term
    bias_train = np.ones((1, np.shape(X_train)[1])) * -1
    bias_test = np.ones((1, np.shape(X_test)[1])) * -1
    
    X_train = np.append(X_train, bias_train, axis = 0)
    X_test = np.append(X_test, bias_test, axis = 0)
    
    return X_train, X_test, y_train, y_test

def minibatch(x_train,y_train,K):
    # Batch Size: K
    # X_train_batches, y_train_batches should be a list of lists of size K.
    x_train_batches = [x_train[i:i+K] for i in range(0, len(x_train), K)]
    y_train_batches = [y_train[i:i+K] for i in range(0, len(y_train), K)]
        
    return x_train_batches, y_train_batches

def initialize_model():
    #TODO (Can use the similar approach used in HW1)
    # e.g. He Initialization for W0-W2, Xavier Initialization for W3
    # Also, initialize your model with a dropout parameter of 0.25 and use_batchnorm being true.
    
    M_0 = 120
    M_1 = 100
    M_2 = 80
    M_3 = 80

    #Initialize the weights. Adding -1 for the bias term at the end of the vector.
    # Random initialization
    W0 = np.random.rand(np.shape(X_train)[0],M_0) # K x M_1 = 785 x 120
    # He initialization
    W1 = np.ones((M_1,M_2)) # 120 x 100
    W1 = np.random.randn(np.shape(W1)[0], np.shape(W1)[1]) * np.sqrt(2/np.shape(W1)[0])

    W2 = np.ones((M_2,M_3)) # 100 x 80
    W2 = np.random.randn(np.shape(W2)[0], np.shape(W2)[1]) * np.sqrt(2/np.shape(W2)[0])

    # Xavier initialization
    W3 = np.random.randn(M_3, 1) * np.sqrt(1/M_3)

    print(f"Size of W0 : {W0.shape}, Size of W1 : {W1.shape}, Size of W2 : {W2.shape}, Size of W3 : {W3.shape}")
    four_layer_nn  = NN4()
    four_layer_nn.layers[0].W = W0
    four_layer_nn.layers[1].W = W1
    four_layer_nn.layers[2].W = W2
    four_layer_nn.layers[3].W = W3

    return four_layer_nn

def train_model(model, X_train_batches, y_train_batches, gamma, beta):
    #TODO : Call async_SGD and sync_SGD to train two versions of the same model. Compare their outcomes and runtime.
    #Update both your models with final updated weights and return them
    model_async = async_sgd(model, X_train_batches, y_train_batches,gamma, beta)
    # model_sync = sync_sgd(model, X_train_batches, y_train_batches)

    return model_async, model_sync

def test_model(model, X_test, y_test):
    '''Tests the accuracy of the neural network.'''
    accuracy = None
    # final_W = model.W
    
    # set the weights in the layers
    # for layer in range(len(model.layers)):
    #     model.layers[layer].W = final_W[layer]
    
    # get the predictions of the algorithm using testing x as the input
    y_hat = model.nn2(X_test)

    # get the number of test instances
    T = np.shape(y_test)[1]

    A = np.absolute(y_test - y_hat)
    
    # check if the value is greater than 0 and set it 1 if so.
    for x in range(np.shape(A)[1]):
            if A[0][x] > 0: 
                A[0][x] = 1

    # calculate the accuracy 
    accuracy = 1/T * np.sum(A)
    
    return accuracy

# %% [markdown]
# ### 5. Run the program

# %%

#load data
x, y = load_dataset()

#prepare data
x, y = prepare_data(x,y)

# split data set
X_train, X_test, y_train, y_test = split_train_test(x,y)

#initialize model
model = initialize_model()

K = 30
x_train_batches, y_train_batches = minibatch(X_train,y_train,K)

#training model
gamma = 0.9
beta = 0.5
model_async, model_sync = train_model(model, x_train_batches, y_train_batches, gamma, beta)
print(f"Completed training, now testing...")   

#testing model
accuracy_async = test_model(model_async, X_test, y_test)
print(f"Completed testing model using asynchronous SGD - Accuracy : {accuracy_async}")   

accuracy_sync = test_model(model_sync, X_test, y_test)
print(f"Completed testing model using synchronous SGD - Accuracy : {accuracy_sync}") 

# %%



