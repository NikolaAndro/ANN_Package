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
from mlcvlab_GPU.models.nn4 import NN4
from mlcvlab_GPU.nn.losses import l2
from mlcvlab_GPU.optim.sgd import SGD
from mlcvlab_GPU.optim.async_sgd import async_sgd
from mlcvlab_GPU.optim.sync_sgd import sync_sgd

# from numba import jit, njit, vectorize, cuda, uint32, f8, uint8


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
    # X_train, X_test = x[:6].T, x[69900:].T
    # y_train, y_test = y[:6].T, y[69900:].T
    
    # adding -1 to the end of every x  as a bias term
    bias_train = np.ones((1, np.shape(X_train)[1])) * -1
    bias_test = np.ones((1, np.shape(X_test)[1])) * -1
    
    X_train = np.append(X_train, bias_train, axis = 0)
    X_test = np.append(X_test, bias_test, axis = 0)
    
    return X_train, X_test, y_train, y_test

def minibatch(x_train,y_train,K):
    # Batch Size: K
    # X_train_batches, y_train_batches should be a list of lists of size K.
    x_train = x_train.T
    y_train = y_train.T

    x_train_batches = np.array([x_train[i:i+K].T for i in range(0, len(x_train), K)])
    y_train_batches = np.array([y_train[i:i+K].T for i in range(0, len(y_train), K)])
 
    return x_train_batches, y_train_batches

def initialize_model(M_0,M_1,M_2,M_3, use_batch_norm = True, dropout_p = 0.5):  
    #Initialize the weights. Adding -1 for the bias term at the end of the vector.
    # Random initialization
    #W0 = np.random.rand(np.shape(X_train)[0],M_0) # K x M_1 = 785 x 120
    W0 = np.ones((np.shape(X_train)[0],M_0)) # K x M_1 = 785 x 120
    W0 = np.random.randn(np.shape(W0)[0], np.shape(W0)[1]) * np.sqrt(2/np.shape(W0)[0])
    
    # He initialization
    W1 = np.ones((M_0,M_1)) # 120 x 100
    W1 = np.random.randn(np.shape(W1)[0], np.shape(W1)[1]) * np.sqrt(2/np.shape(W1)[0])

    W2 = np.ones((M_1,M_2)) # 100 x 80
    W2 = np.random.randn(np.shape(W2)[0], np.shape(W2)[1]) * np.sqrt(2/np.shape(W2)[0])

    # Xavier initialization
    W3 = np.random.randn(M_2, M_3) * np.sqrt(1/M_2)

    print(f"Size of W0 : {W0.shape}, Size of W1 : {W1.shape}, Size of W2 : {W2.shape}, Size of W3 : {W3.shape}")
    four_layer_nn  = NN4(use_batchnorm=use_batch_norm, dropout_param=dropout_p)
    four_layer_nn.layers[0].W = W0
    four_layer_nn.layers[1].W = W1
    four_layer_nn.layers[2].W = W2
    four_layer_nn.layers[3].W = W3

    four_layer_nn.dropout_param = dropout_p

    return four_layer_nn

def train_model(model, x_train_batches, y_train_batches, num_epochs=100, learning_rate=0.1):
    #TODO : Call async_SGD and sync_SGD to train two versions of the same model. Compare their outcomes and runtime.
    #Update both your models with final updated weights and return them
    # model_async = async_sgd(model, x_train_batches, y_train_batches,R = num_epochs, lr = learning_rate)
    model_sync = sync_sgd(model, x_train_batches, y_train_batches,R = num_epochs, lr = learning_rate)
    model_async = None
    return model_async, model_sync

def test_model(model, X_test, y_test):
    '''Tests the accuracy of the neural network.'''
    accuracy = None
    # final_W = model.W
    
    # set the weights in the layers
    # for layer in range(len(model.layers)):
    #     model.layers[layer].W = final_W[layer]
    
    # get the number of test instances
    T = np.shape(y_test)[1]
    
    # get the predictions of the algorithm using testing x as the input
    
    y_hat = np.zeros(np.shape(y_test))
    
    for index, image in enumerate(X_test.T):
        y_hat[0][index] = model.nn4(image,'test')

    

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
# promena samo da ti pokazem kako se radi sa gitom
# u visual studio

#load data
x, y = load_dataset()

#prepare data
x, y = prepare_data(x,y)

# split data set
X_train, X_test, y_train, y_test = split_train_test(x,y)

#initialize model
M_1 = 120
M_2 = 100
M_3 = 80
M_4 = 1 # Layer 4 must be 1 since this is a binary classification problem
dropout_p_val = 0.5

model = initialize_model(M_1,M_2,M_3,M_4, use_batch_norm = False, dropout_p = dropout_p_val)

K = 6000
x_train_batches, y_train_batches = minibatch(X_train,y_train,K)

#set values for training model

num_epochs = 1
learning_rate = 0.1
model_async, model_sync = train_model(model, x_train_batches, y_train_batches, num_epochs=num_epochs, learning_rate=learning_rate)
print(f"Completed training, now testing...")   

#testing model
accuracy_async = test_model(model_async, X_test, y_test) * 100
print(f"Completed testing model using asynchronous SGD - Accuracy : {accuracy_async}%")   

#accuracy_sync = test_model(model_sync, X_test, y_test)
#print(f"Completed testing model using synchronous SGD - Accuracy : {accuracy_sync}") 

# %%



