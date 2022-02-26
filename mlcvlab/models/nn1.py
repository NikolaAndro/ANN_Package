import numpy as np
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import sigmoid, sigmoid_grad
from .base import Layer


class NN1():
    def __init__(self):
        self.layers = [
            Layer(None, sigmoid)]
        self.W = None

    def nn1(self, x):
        #TODO
        raise NotImplementedError("NN1 model not implemented")

    def grad(self, x, y, W):
        # TODO
        raise NotImplementedError("NN1 gradient (backpropagation) not implemented")

    def emp_loss_grad(self, train_X, train_y, W, layer):
        # emp_loss_ = 0
        # emp_loss_grad_ = 0
        
        # TODO
        # return emp_loss_grad_
        raise NotImplementedError("NN1 Emperical Loss grad not implemented")
       