# No additional 3rd party external libraries are allowed
from random import betavariate
from re import M
import numpy as np
from regex import E


def batchnorm(z_tilda, gamma, beta, eps, mode = 'test'):
    '''
    Input:
    - z_tilda: Data of shape (M, 1)
    - gamma: Scale parameter of shape (1,)
    - beta: Shift paremeter of shape (1,)
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
    running_mean = np.zeros(z_tilda.shape()[1], dtype=z_tilda.dtype)
    running_variance = np.zeros(z_tilda.shape()[1], dtype=z_tilda.dtype)

    if mode == 'train':

        # Find the mean of the minibatch
        z_tilda_mean = z_tilda.mean(axis = 0)

        # Find the variance of the minibatch
        z_tilda_variance = z_tilda.var(axis=0)

        # Find the running mean and variance for the backward propagation
        running_mean = momentum * running_mean + (1  - momentum) * z_tilda_mean
        running_variance = momentum * running_variance + (1  - momentum) * z_tilda_variance


        # Normalize the layer
        z_tilda_centered = z_tilda - z_tilda_mean # dim: M x 1
        standard = np.sqrt(z_tilda_variance + eps) # dim: M x 1
        z_tilda_normalized = z_tilda_centered / standard # dim: M x 1

        # Scale and shift using gamma and beta
        output_batch = gamma * z_tilda_normalized + beta # dim: M x 1

        # Keep all this information in memory to be used for backprop
        cache = (z_tilda_normalized, z_tilda_mean, z_tilda_variance, gamma, beta, running_mean, running_variance)
    
    elif mode =="test":
        z_tilda_normalized = (z_tilda - running_mean) / np.sqrt(running_variance + eps)
        output_batch = gamma + z_tilda_normalized + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


    return output_batch, cache


def batchnorm_grad(grad_l_wrt_y, input_x, eps, cache):
    """
    Backward pass for batch normalization.
    Inputs:
    - grad_l_wrt_y:  This is glrad_l_wrt_b in this nn architecture; dim: M x 1
    - input_x: input to the layer
    - cache: x_normalized, x_mean, x_variance, gamma, beta, running_mean, running_variance- from cache in forward batchnorm function.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    # num features
    m = grad_l_wrt_y.shape[0]

    # get the variables from forward pass in cache 
    x_normalized, x_mean, x_variance, gamma, _, __,___ = cache

    # Equation 1 from the backprop algorithm  from the original paper of batchnorm
    # partial derivative of L with respect to x_hat
    pd_l_wrt_xhat = grad_l_wrt_y * gamma # dim: M x 1

    # partial derivative of L with respect to gamma
    # pd (partial derivative)
    pd_l_wrt_gamma =  (grad_l_wrt_y * x_normalized).sum(axis=0) # dim: 1 x 1 

    #partial derivative of l wrt beta
    pd_l_wrt_beta = grad_l_wrt_y.sum(axis=0) # dim: 1 x 1

    # partial derivativev of l wrt mean
    pd_l_wrt_mean = (pd_l_wrt_xhat * (-1 / np.sqrt(x_variance + eps)) ).sum(axis=0) # dim: 1 x 1

    # partial derivative of l wrt variance
    pd_l_wrt_varance =  (pd_l_wrt_xhat * (input_x - x_mean) * (-0.5) * (x_variance + eps)**(-3/2)).sum(axis=0) # dim: 1 x 1

    # partial derivative of l wrt x
    pd_l_wrt_x = pd_l_wrt_xhat * 1/np.sqrt(x_variance + eps) + pd_l_wrt_varance * \
        (2*(input_x - x_mean) / m) + pd_l_wrt_mean * 1/m # dim: M x 1

    return pd_l_wrt_x, pd_l_wrt_gamma, pd_l_wrt_beta




    