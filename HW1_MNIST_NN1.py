# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

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

# get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# %% [markdown]
# ### 2. Import from mlcblab

# %%
from mlcvlab.models.nn1 import NN1
from mlcvlab.models.nn2 import NN2
from mlcvlab.nn.losses import l2
from mlcvlab.optim.sgd import SGD
from mlcvlab.optim.adam import Adam

# %% [markdown]
# ### 3. Set Seed

# %%
np.random.seed(42)

# %% [markdown]
# ### 4.Example code to load data and view image

# %%
def display_digit(x, index):
    some_digit = x[0]
    some_digit_image = some_digit.reshape((28, 28))

    plt.imshow(some_digit_image, cmap='binary')
    plt.axis('off')
    plt.show()


# %%
#x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)


# %%
#display_digit(x, 0)


# %%
#y

# %% [markdown]
# ### 5. Helper functions

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


def initialize_model(X_train, X_test, y_train, y_test):
    '''Setting up the size of the weights/layer vectors. W0 is M x K shape where M = arbitrary number and K = 785 and W1 is M x 1 which is 60000 x 1 '''

    # multiple ways to initialize the weights
    
    # Initialization with ones
    # W = np.ones((np.shape(X_train)[0],1)) 
    
    # Random Initialization
    # W = np.random.rand(np.shape(X_train)[0], 1)
    
    # Xavier initialization
    W = np.random.randn(np.shape(X_train)[0], 1) * np.sqrt(1/np.shape(X_train)[0])
    
    
    # set the last term to -1 as bias term
    W[-1] = -1
    
    one_layer_nn  = NN1()
    one_layer_nn.W = W
    one_layer_nn.layers[0].W = W

    return one_layer_nn

def train_model(model, X_train, y_train):
    '''Training the model using SGD or Adam optimizer.'''
    final_W = SGD(model, X_train, y_train, lr=0.1)
    # final_W = Adam(model, X_train, y_train)

    return final_W

def test_model(model, X_test, y_test, final_W):
    '''Tests the accuracy of the neural network.'''
    accuracy = None
    model.W = final_W[0]
    
    # get the predictions of the algorithm using testing x as the input
    y_hat = model.nn1(X_test)

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
# ### 6. Run the program

# %%
warnings.filterwarnings("ignore")

#load data
x, y = load_dataset()

#prepare data
x, y = prepare_data(x,y)

# split data set
X_train, X_test, y_train, y_test = split_train_test(x,y)

#initialize model
model = initialize_model(X_train, X_test, y_train, y_test)

#training model
final_W = train_model(model, X_train, y_train)
print(f"Completed training model - final W :\n {final_W}")


#testing model
accuracy = test_model(model, X_test, y_test, final_W)
percentage = 100 * accuracy
print(f"Completed testing model (nn1) - Accuracy : {percentage:2.1f}%")    


# %%



