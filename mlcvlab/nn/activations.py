# No additional 3rd party external libraries are allowed
from tkinter import Y
import numpy as np

def relu(x):
     # TODO
    y = 0 if x < 0 else x
    return y
    # raise NotImplementedError("ReLU function not implemented")

def relu_grad(z):
     # TODO
    y_prime = 0 if z < 0 else 1
    return y_prime
    # raise NotImplementedError("Gradient of ReLU function not implemented")

def sigmoid(x):
    # TODO
    y = 1/(1 + np.exp(-x))
    return y
    # raise NotImplementedError("Sigmoid function not implemented")

def sigmoid_grad(z):
    # TODO
    y_prime = sigmoid(z) * (1 - sigmoid(z))
    return y_prime
    # raise NotImplementedError("Gradient of Sigmoid function not implemented")

def softmax(x):
    # TODO
    raise NotImplementedError("Softmax function not implemented")

def softmax_grad(z):
    # TODO
    raise NotImplementedError("Gradient of Softmax function not implemented")

def tanh(x):
    # TODO
    y = (2/(1 + np.exp(-2*x)))-1
    return y
    # raise NotImplementedError("Tanh function not implemented")

def tanh_grad(z):
    # TODO
    y = tanh(z)
    y_prime = 1 - y**2
    return y_prime

    # raise NotImplementedError("Gradient of Tanh function not implemented")
