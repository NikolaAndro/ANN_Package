# No additional 3rd party external libraries are allowed
from re import M
import numpy as np

def linear(x, W):
    # TODO
    y = W @ x
    print("linear: ",y)
    return y
    # raise NotImplementedError("Linear function not implemented")
    
def linear_grad(x):
    # TODO
    print("linear_grad: ", x)
    return x
    # raise NotImplementedError("Gradient of Linear function not implemented")

def radial(x, W):
    # TODO
    y = np.sqrt(np.sum((x - W)**2))
    # y = np.sqrt(np.matmul((x-W),(x-W).T))
    print("radial: ",y)
    return y
    # raise NotImplementedError("Radial Basis function not implemented")
    
def radial_grad(x, W):
    # TODO
    # delta_y = -2 * loss_grad_y * (x - W)
    # delta_y = ((x-W)/loss_grad_y)
    y = radial(x,W)
    delta_y = (x-W)/y
    print("radial_grad: ", delta_y)
    return delta_y
    # raise NotImplementedError("Gradient of Radial Basis function not implemented")