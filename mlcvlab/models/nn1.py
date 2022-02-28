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
        y = sigmoid(np.dot(self.W, x))
        return y
        # raise NotImplementedError("NN1 model not implemented")

    def grad(self, x, y, W):
        # TODO
        z = np.dot(W,x)
        y_hat = sigmoid(z)

        #compute the parts for the final gradient
        del_z_l = np.dot(l2_grad(y,y_hat),sigmoid_grad(z))

        del_w_l = np.dot(del_z_l,x)

        return del_w_l

        # raise NotImplementedError("NN1 gradient (backpropagation) not implemented")

    def emp_loss_grad(self, train_X, train_y, W, layer):
        # emp_loss_ = 0
        # emp_loss_grad_ = 0
        
        # TODO
        # return emp_loss_grad_
        raise NotImplementedError("NN1 Emperical Loss grad not implemented")
       