# No additional 3rd party external libraries are allowed
import numpy as np
from numpy.linalg import norm 

def l2(y, y_hat):
    """L2 Loss Function is used to minimize the error 
    which is the sum of the all the squared differences between 
    the true value and the predicted value."""
    # This is MSE - mean squared error loss function
    # Norm Function
    z = np.sqrt(np.sum((y-y_hat)**2))
    return z

def l2_grad(y, y_hat):
    """Gradient of l2 loss function"""
    z = l2(y,y_hat)
    delta_z = (1/z)*(y-y_hat)
    return delta_z

def cross_entropy(y, y_hat):
    '''Calculates binary cross entropy. '''
    # Calculate the basic equation for cross entropy
    return  -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)   

def cross_entropy_grad(y, y_hat):
    delta_z = ((1-y)/(1-y_hat))-(y/y_hat)
    return delta_z    