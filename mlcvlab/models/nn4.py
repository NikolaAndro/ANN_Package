import numpy as np
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer
from mlcvlab.nn.dropout import dropout, dropout_grad


class NN4():
    def __init__(self, use_batchnorm=False, dropout_param=0):
        self.layers = [
            Layer(None, relu),
            Layer(None, relu),
            Layer(None, relu),
            Layer(None, sigmoid)]
        
        self.use_batchnorm = use_batchnorm

        #used in dropout implementation
        self.dropout_param = dropout_param

    def nn4(self, x):
        # TODO
        if self.use_batchnorm:
            raise NotImplementedError("NN4 Batchnorm model not implemented")

        else:
            raise NotImplementedError("NN4 Without Batchnorm model not implemented")

    def grad(self, x, y):
        # TODO  
        if self.use_batchnorm:
            raise NotImplementedError("NN4 gradient (backpropagation) Batchnorm model not implemented")

        else:
            raise NotImplementedError("NN4 gradient (backpropagation) Without Batchnorm model not implemented")    

    def emp_loss_grad(self, train_X, train_y, layer):
        # emp_loss_ = 0
        # emp_loss_grad_ = None
        # TODO
        raise NotImplementedError("NN4 Emperical Loss grad not implemented")