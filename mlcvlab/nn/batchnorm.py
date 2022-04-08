# No additional 3rd party external libraries are allowed
from random import betavariate
from re import M
import numpy as np
from regex import E


def batchnorm(x, gamma, beta, eps, mode = 'test'):
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
        standard = np.sqrt(x_variance + eps)
        x_normalized = x_centered / standard

        # Scale and shift using gamma and beta
        output_x = gamma * x_normalized + beta

        # Keep all this information in memory to be used for backprop
        cache = (x_normalized, x_mean, x_variance, gamma)
    
    elif mode =="test":
        x_normalized = (x - running_mean) / np.sqrt(running_variance + eps)
        output_x = gamma + x_normalized + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


    return output_x, cache


def batchnorm_grad(grad_l_wrt_y, input_x, eps, cache):
    """
    Backward pass for batch normalization.
    Inputs:
    - grad_l_wrt_y: gradient of the loss function
    - input_x: input to the layer
    - cache: x_normalized, x_mean, x_variance, gamma - from cache in forward batchnorm function.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    # num features
    m = grad_l_wrt_y.shape[0]

    # get the variables from forward pass in cache 
    x_normalized, x_mean, x_variance, gamma = cache

    # Equation 1 from the backprop algorithm  from the original paper of batchnorm
    # partial derivative of L with respect to x_hat
    pd_l_wrt_xhat = grad_l_wrt_y * gamma

    # partial derivative of L with respect to gamma
    # pd (partial derivative)
    pd_l_wrt_gamma =  (grad_l_wrt_y * x_normalized).sum(axis=0)

    #partial derivative of l wrt beta
    pd_l_wrt_beta = grad_l_wrt_y.sum(axis=0)

    # partial derivativev of l wrt mean
    pd_l_wrt_mean = (pd_l_wrt_xhat * (-1 / np.sqrt(x_variance + eps)) ).sum(axis=0)

    # partial derivative of l wrt variance
    pd_l_wrt_varance =  (pd_l_wrt_xhat * (input_x - x_mean) * (-0.5) * (x_variance + eps)**(-3/2)).sum(axis=0)

    # partial derivative of l wrt x
    pd_l_wrt_x = pd_l_wrt_xhat * 1/np.sqrt(x_variance + eps) + pd_l_wrt_varance * \
        (2*(input_x - x_mean) / m) + pd_l_wrt_mean * 1/m

    return pd_l_wrt_x, pd_l_wrt_gamma, pd_l_wrt_beta




    