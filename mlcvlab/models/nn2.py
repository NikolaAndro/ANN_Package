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
        '''Function returns value for y_hat = sigmoid(w2.T * sigmoid(w1.T * x)).'''
        y_hat_1 = relu(np.dot(self.layers[0].W , x))
        y_hat_2 = sigmoid(np.dot(self.layers[1].W.T,y_hat_1))

        return y_hat_2


    def grad(self, x, y, W):
        '''Returns a gradient for nn2 as a tuple of grad_l_wrt_w1 and grad_l_wrt_w2.'''

       
        # set up the size of the dimensions. Storing in one variable not to have to retrieve it multiple times.
        M = np.shape(self.layers[0].W)[0]
        K = np.shape(self.layers[0].W)[1]
        # Set up
        z_1 = np.dot(self.layers[0].W,x)
        z_1_tilda = relu(z_1)
        z_2 = np.dot(self.layers[1].W.T, z_1_tilda) # 1 x 1
        y_hat = sigmoid(z_2)
        

        #********* calculate the outter layer gradient **********
        
        # Following the equations from the PDF doccument. 
        grad_y_hat_wrt_z2 = sigmoid_grad(z_2)
        grad_l_wrt_y_hat = l2_grad(y,y_hat)
        grad_l_wrt_z2 = np.dot(grad_l_wrt_y_hat, grad_y_hat_wrt_z2)
        
        
        grad_z2_wrt_z1_tilda = self.layers[1].W # 120 x 1
        grad_l_wrt_z1_tilda = np.dot(np.asscalar(grad_l_wrt_z2),grad_z2_wrt_z1_tilda) # 120 x 1
        
        # For every element in z_1_tilda there is going to be an element z_i in z so the total will
        # end up being 1 x M size
        grad_z1_tilda_wrt_z1 = np.zeros((M,M)) # 120 x 120
        for i in range(np.shape(z_1_tilda)[0]):
            z_ij = np.dot(int(z_1_tilda[i]), z_1)
            grad_z1_tilda_wrt_z1[int(i)] = z_ij.T
        grad_l_wrt_z1 = np.dot(grad_l_wrt_z1_tilda.T, grad_z1_tilda_wrt_z1) # 1 x 120
        
        
        grad_z1_wrt_w1 = np.zeros((M,M,K)) # 120 x 120 x 784        
        # For every element in z_1 there is going to be a matrix W_ij (M x K size) so the total will
        # end up being M x M x K size
        for i in range(np.shape(z_1)[0]):
            z_i_W_ij = np.dot(int(z_1[i]), self.layers[0].W)
            grad_z1_wrt_w1[int(i)] = z_i_W_ij
        
        grad_l_wrt_w1 = np.dot(grad_l_wrt_z1, grad_z1_wrt_w1)  
        
        # reshape the dimentsions from 1 x M x K to just M x K
        grad_l_wrt_w1 = grad_l_wrt_w1.reshape(M,K)
        
        #********* calculate the inner layer gradient **********

        grad_z2_wrt_w2 = z_2 * self.layers[1].W
        grad_l_wrt_w2 = int(grad_l_wrt_z2.T) * grad_z2_wrt_w2
        
        return grad_l_wrt_w1, grad_l_wrt_w2.T

    def emp_loss_grad(self, train_X, train_y, W, radnom_W_index):
        '''Calculates the gradient of empirical loss function for NN1.'''
        # number of iterations
        N = np.shape(train_X)[1]
       
        # replace the randomly picked W with one-hot vector
        self.layers[radnom_W_index].W = W
        self.W[radnom_W_index]=W
        sum_img_emp_loss = [np.zeros((np.shape(self.layers[0].W)[0], np.shape(self.layers[0].W)[1])),np.zeros((np.shape(self.layers[1].W)[1],np.shape(self.layers[1].W)[0]))]
        
        a = 0
        #get the empirical loss image by image
        for tx, ty in zip(train_X.T,train_y.T):
            emp_loss = self.grad( tx, ty, W)
            sum_img_emp_loss[0] += emp_loss[0]
            sum_img_emp_loss[1] += emp_loss[1]
            a = a + 1
        
        sum_img_emp_loss[1] = sum_img_emp_loss[1].T
        # calculate the mean for both layers
        emp_loss_grad = []
        emp_loss_grad.append( (1 / N) * sum_img_emp_loss[0])
        emp_loss_grad.append( (1 / N) * sum_img_emp_loss[1])
        
        return emp_loss_grad