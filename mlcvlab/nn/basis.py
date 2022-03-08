# No additional 3rd party external libraries are allowed
from re import M
import numpy as np

def linear(x, W):
    '''Linear basis.'''
    y = W @ x
    return y
    
def linear_grad(x):
    '''Gradient of linear basis.'''
    return x

def radial(x, W):
    '''Radial basis.'''
    y = np.sum((x - W)**2)
    print("radial: ",y)
    return y
    
def radial_grad(loss_grad_y, x, W):
    '''Gradient of radial basis.'''
    y = radial(x,W)
    delta_y = (x-W)/loss_grad_y
    # delta_y = -2 * loss_grad_y * (x-W)
    delta_y = delta_y.flatten()
    print("Delta_radial: \n", delta_y)
    return delta_y
