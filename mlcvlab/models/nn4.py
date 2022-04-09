import numpy as np
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer
from mlcvlab.nn.dropout import dropout, dropout_grad
from mlcvlab.nn.batchnorm import batchnorm, batchnorm_grad

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

    def nn4(self, x, curent_mode):
        # TODO
        if self.use_batchnorm:
            #************************** LAYER 1 ************************
            z_1_tilda = relu(np.dot(self.layers[0].W , x))
            # apply batchnorm
            b_1 = batchnorm(z_1_tilda, gamma=0.1, beta=0.1, eps=0.001, mode=curent_mode )
            # apply dropout
            y_1 = dropout(b_1, p = 0.5, mode = curent_mode)

            #************************** LAYER 2 ************************
            z_2_tilda = relu(np.dot(self.layers[1].W , y_1))
            # apply batchnorm
            b_2 = batchnorm(z_2_tilda, gamma=0.1, beta=0.1, eps=0.001, mode=curent_mode )
            # apply dropout
            y_2 = dropout(b_2, p = 0.5, mode = curent_mode)

            #************************** LAYER 3 ************************
            z_3_tilda = relu(np.dot(self.layers[2].W , y_2))
            # apply batchnorm
            b_3 = batchnorm(z_3_tilda, gamma=0.1, beta=0.1, eps=0.001, mode=curent_mode )
            # apply dropout
            y_3 = dropout(b_3, p = 0.5, mode = curent_mode)

            #************************** LAYER 4 ************************
            y_hat = sigmoid(np.dot(self.layers[3].W , y_3))


        else:
            #************************** LAYER 1 ************************
            z_1_tilda = relu(np.dot(self.layers[0].W , x))
            # apply dropout
            y_1 = dropout(z_1_tilda, p = 0.5, mode = curent_mode)

            #************************** LAYER 2 ************************
            z_2_tilda = relu(np.dot(self.layers[1].W , y_1))
            # apply dropout
            y_2 = dropout(z_2_tilda, p = 0.5, mode = curent_mode)

            #************************** LAYER 3 ************************
            z_3_tilda = relu(np.dot(self.layers[2].W , y_2))
            # apply dropout
            y_3 = dropout(z_3_tilda, p = 0.5, mode = curent_mode)

            #************************** LAYER 4 ************************
            y_hat = sigmoid(np.dot(self.layers[3].W , y_3))

        return y_hat

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