# No additional 3rd party external libraries are allowed
from random import betavariate
import numpy as np
from regex import E


def batchnorm(x, gamma, beta, mode = 'train'):
    '''
    Input:
    - x: Data of shape (M, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - mode: 'train' or 'test'; required
    output:
    - output_batch and the rest of the variables for backprop

    Normalizes the values in every layer then scales and shifts them 
    to create a new distribution at each layer intstead of zero mean and 
    unit variance. We use gamma and beta to scale and shift the values. 
    This scaling and shifting makes sure that data is not the same in every 
    layer and it also helps in regularizing the network since both gamma and 
    beta add's some random noise to the data.
    '''
    
    # Hardcode the momentum
    momentum = 0.9

    # Initialize running variance and mean for backprop
    running_mean = np.zeros(x.shape()[1], dtype=x.dtype)
    running_variance = np.zeros(x.shape()[1], dtype=x.dtype)

    if mode == 'train':

        # Find the mean of the minibatch
        x_mean = x.mean(axis = 0)

        # Find the variance of the minibatch
        x_variance = x.var(axis=0)

        # Find the running mean and variance for the backward propagation
        running_mean = momentum * running_mean + (1  - momentum) * x_mean
        running_variance = momentum * running_variance + (1  - momentum) * x_variance


        # Normalize the layer
        x_centered = x - x_mean
        standard = np.sqrt(x_variance + np.e)
        x_normalized = x_centered / standard

        # Scale and shift using gamma and beta
        output_batch = gamma * x_normalized + beta

        # Keep all this information in memory to be used for backprop
        cache = (x_normalized, x_centered, x_mean, x_variance, standard, gamma, beta)
    
    elif mode =="test":
        x_normalized = (x - running_mean) / np.sqrt(running_variance + np.e)
        output_batch = gamma + x_normalized + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


    return output_batch, cache

def batchnorm_grad(y, x, gamma, beta, eps):
    # TODO
    raise NotImplementedError("Gradiant of Batchnorm Not Implemented")