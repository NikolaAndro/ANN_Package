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
    
def radial_grad(x, W):
    '''Gradient of radial basis.'''
    delta_y = 2*(x-W)**2   
    return delta_y
