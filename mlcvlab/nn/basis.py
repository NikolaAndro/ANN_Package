# No additional 3rd party external libraries are allowed
from re import M
import numpy as np

def linear(x, W):
    y = W @ x
    # print("linear: ",y)
    return y
    
def linear_grad(x):
    # print("linear_grad: ", x)
    return x

def radial(x, W):
    # y = np.sqrt(np.sum((x - W)**2))
    y = np.sum((x - W)**2)
    print("radial: ",y)
    return y
    
def radial_grad(loss_grad_y, x, W):
    y = radial(x,W)
    # delta_y = (x-W)/loss_grad_y
    delta_y = -2 * loss_grad_y * (x-W)
    delta_y = delta_y.flatten()
    print("Delta_radial: \n", delta_y)
    return delta_y
