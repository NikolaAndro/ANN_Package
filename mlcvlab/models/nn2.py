import numpy as np
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer


class NN2():
    def __init__(self):
        self.layers = [
            Layer(None, relu), 
            Layer(None, sigmoid)]

    def nn2(self, x):
        # TODO
        raise NotImplementedError("NN2 model not implemented")

    def grad(self, x, y, W):
        # TODO
        raise NotImplementedError("NN2 gradient (backpropagation) not implemented")        

    def emp_loss_grad(self, train_X, train_y, W, layer):
        # emp_loss_ = 0
        # emp_loss_grad_ = None
        # TODO

        # return emp_loss_grad_
        raise NotImplementedError("NN2 Emperical Loss grad not implemented")