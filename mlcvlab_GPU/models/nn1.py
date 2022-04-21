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
        '''Function returns value for z = sigmoid(w.T * x).\
        In pytorch this is model.forward()'''
        return sigmoid(np.dot(self.W.T, x))

    def grad(self, x, y, W):
        '''Returns a gradient of the function.\
        In pytorch this is model.backwards()'''
        
        # set up
        #Transpose because for image in trainx dataset we will have 1 real number as Z, so we will have 60000 x 1 for z
        z = (np.dot(W.T,x))
        y_hat = (sigmoid(z))
        
        grad_l_wrt_y_hat = l2_grad(y,y_hat)
        grad_y_hat_wrt_z = (sigmoid_grad(z))
        
        #compute the parts for the final gradient
        grad_l_wrt_z = grad_l_wrt_y_hat * grad_y_hat_wrt_z
    
        grad_l_wrt_w =  grad_l_wrt_z * x
        
        return grad_l_wrt_w

    def emp_loss_grad(self, train_X, train_y, W, layer):
        '''Calculates the gradient of empirical loss function for NN1.'''
        # number of iterations
        N = np.shape(train_X)[1]
        loss_gradient = self.grad(train_X,train_y,W) 
        emp_loss_grad = (1 / N) * np.sum(loss_gradient,axis=1) 
        # reshape the vector
        emp_loss_grad = np.reshape(emp_loss_grad,(np.shape(emp_loss_grad)[0],1))
        return emp_loss_grad
       