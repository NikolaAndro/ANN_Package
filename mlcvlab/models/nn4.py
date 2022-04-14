import numpy as np
from mlcvlab.nn.losses import l2, l2_grad
from mlcvlab.nn.basis import linear, linear_grad
from mlcvlab.nn.activations import relu, sigmoid, sigmoid_grad, relu_grad
from .base import Layer
from mlcvlab.nn.dropout import dropout, dropout_grad
from mlcvlab.nn.batchnorm import batchnorm, batchnorm_grad
from numba import jit, njit, vectorize, cuda, uint32, f8, uint8

class NN4():
    def __init__(self, use_batchnorm=False, dropout_param=0):
        self.layers = [
            Layer(None, relu, None, None, None, None ),
            Layer(None, relu, None, None, None, None),
            Layer(None, relu, None, None, None, None),
            Layer(None, sigmoid, None, None, None, None)]
        
        self.use_batchnorm = use_batchnorm

        #used in dropout implementation
        self.dropout_param = dropout_param

    def nn4(self, x, curent_mode):
        # TODO
        if self.use_batchnorm:
            #************************** LAYER 1 ************************
            z_1 = np.dot(self.layers[0].W , x)
            self.layers[0].z = z_1

            z_1_tilda = relu(z_1)
            self.layers[0].z_tilda = z_1_tilda
            # apply batchnorm
            batch_cache = self.layers[0].batch_norm[2]
            b_1 = batchnorm(z_1_tilda, gamma=0.1, beta=0.1, eps=0.001, mode=curent_mode )
            self.layers[0].batch_norm = b_1
            # apply dropout
            y_1 = dropout(b_1[0], p = 0.5, mode = curent_mode) # results in tuple (b_drop, p, mode, mask)

            self.layers[0].y_out = y_1[0]

            #************************** LAYER 2 ************************
            z_2 = np.dot(self.layers[1].W , y_1)
            self.layers[1].z  = z_2

            z_2_tilda = relu(z_2)
            self.layers[1].z_tilda = z_2_tilda
            # apply batchnorm
            b_2 = batchnorm(z_2_tilda, gamma=0.1, beta=0.1, eps=0.001, mode=curent_mode )
            self.layers[1].batch_norm = b_2
            # apply dropout
            y_2 = dropout(b_2[0], p = 0.5, mode = curent_mode)
            self.layers[1].y_out = y_2[0]

            #************************** LAYER 3 ************************
            z_3 = np.dot(self.layers[2].W , y_2)
            self.layers[2].z = z_3

            z_3_tilda = relu(z_3)
            self.layers[2].z_tilda = z_3_tilda
            # apply batchnorm
            b_3 = batchnorm(z_3_tilda, gamma=0.1, beta=0.1, eps=0.001, mode=curent_mode )
            self.layers[2].batch_norm = b_3
            # apply dropout
            y_3 = dropout(b_3[0], p = 0.5, mode = curent_mode)
            self.layers[2].y_out = y_3[0]

            #************************** LAYER 4 ************************
            z_4 = np.dot(self.layers[3].W , y_3)
            self.layers[3].z = z_4
            
            y_hat = sigmoid(z_4)
            self.layers[3].y_out = y_hat # just to keep the value in memory for the grad funciton

        else:
            #************************** LAYER 1 ************************
            z_1 = np.dot(self.layers[0].W , x)
            self.layers[0].z = z_1

            z_1_tilda = relu(z_1)
            self.layers[0].z_tilda = z_1_tilda
            # apply dropout
            y_1 = dropout(z_1_tilda, p = 0.5, mode = curent_mode)
            self.layers[0].y_out = y_1[0]

            #************************** LAYER 2 ************************
            z_2 = np.dot(self.layers[1].W , y_1)
            self.layers[1].z  = z_2

            z_2_tilda = relu(z_2)
            self.layers[1].z_tilda = z_2_tilda
            # apply dropout
            y_2 = dropout(z_2_tilda, p = 0.5, mode = curent_mode)
            self.layers[1].y_out = y_2[0]

            #************************** LAYER 3 ************************
            z_3 = np.dot(self.layers[2].W , y_2)
            self.layers[2].z = z_3

            z_3_tilda = relu(np.dot(self.layers[2].W , z_3))
            self.layers[2].z_tilda = z_3_tilda
            # apply dropout
            y_3 = dropout(z_3_tilda, p = 0.5, mode = curent_mode)
            self.layers[2].y_out = y_3[0]

            #************************** LAYER 4 ************************
            z_4 = np.dot(self.layers[3].W , y_3)
            self.layers[3].z = z_4
            y_hat = sigmoid(z_4)
            self.layers[3].y_out = y_hat # just to keep the value in memory for the grad funciton

        return y_hat

    def get_grid_dim(self, minibatch_x, minibatch_y):
        '''Returns the size of the CUDA grid assuming that each block is 16 x 32 size. '''
        return int(np.ceil([minibatch_x/16])[0]), int(np.ceil([minibatch_y/32])[0])


    def layer_4_grad(self,z_4,y,y_hat):
        '''Computes and returns the gradient for the 4th (last) layer.'''
        grad_y_hat_wrt_z4 = sigmoid_grad(z_4)
        grad_l_wrt_y_hat = l2_grad(y,y_hat)
        grad_l_wrt_z4 = np.dot(grad_l_wrt_y_hat, grad_y_hat_wrt_z4) # 1 x 1

        grad_z4_wrt_y3 = self.layers[3].W # returning M3 x M4 matrix = 80 x 1
        grad_l_wrt_y3 = int(grad_l_wrt_z4) * grad_z4_wrt_y3 # M3 x M4 = 80 x 1

        grad_z4_wrt_w4 = self.layers[2].y_out # y3 - dim: M3 x M4 => M3 x 1 since M4 = 1
        grad_l_wrt_w4 = int(grad_l_wrt_z4) * grad_z4_wrt_w4

        # Returning grad_l_wrt_y3 transpose just so we reduce the computation time
        # for every next call of grad_l_wrt_y3 that should be transpose.
        # e.g. in layer_3_grad, layer_2_grad , layer_1_grad
        return grad_l_wrt_w4, grad_l_wrt_y3.T

    def layer_3_grad(self,grad_l_wrt_y3, eps):
        '''Computes and returns the gradient for the 3rd layer.'''
        y3 = self.layers[2].y_out
        mask_3 = self.layers[2].y_out[3] # result of dropout layer is tuple (z_drop, p, mode, mask)
        grad_y3_wrt_b3 = dropout_grad(y3,mask_3) # M3 x M3 diagonal matrix
        # grad_l_wrt_y3 is already transposed when returned from layer_4_grad()
        # grad_l_wrt_y3 => 1 x M3
        # grad_y3_wrt_b3 => M3 x M3
        grad_l_wrt_b3 = np.dot(grad_l_wrt_y3, grad_y3_wrt_b3) # dim: 1 x M3

        #******** updates for batch norm gamma and beta *******
        grad_b3_wrt_z3_tilda, grad_gamma_3, grad_beta_3 = batchnorm_grad(grad_l_wrt_b3,self.layers[2].z_tilda,eps,self.layers[2].batch_norm[1]) #dim: M3 x M3
        grad_l_wrt_z3_tilda = np.dot(grad_l_wrt_b3,grad_b3_wrt_z3_tilda) # dim: 1 x M3


        grad_z3_tilda_wrt_z3 =  relu_grad(self.layers[2].z_tilda)   # M3 x 1
        grad_z3_tilda_wrt_z3 = grad_z3_tilda_wrt_z3 *  np.identity(np.shape(grad_z3_tilda_wrt_z3)[0])# diagonalize to amke it M3 x M3
        grad_l_wrt_z3 = np.dot(grad_l_wrt_z3_tilda,grad_z3_tilda_wrt_z3) # 1 x M3 . M3 x M3 => 1 x M3


        grad_z3_wrt_y2  = self.layers[2].W.T  # dim: M3 x M2
        grad_l_wrt_y2 =  np.dot(grad_l_wrt_z3,grad_z3_wrt_y2) # dim: 1 x M3 . M3 x M2 => 1 x M2 
        

        grad_z3_wrt_w3 = [] # dim: M3 x M2 x M3
        for z in self.layers[2].z:
            grad_z3_wrt_w3.append(np.dot(z,self.layers[2].W))

        grad_l_wrt_w3 = np.dot(grad_l_wrt_z3,grad_z3_wrt_w3)    # dim: 1 x M3 .  M3 x M2 x M3=> 1 x M2 x M3
        # reshape the dimentsions from 1 x M2 x M3 to just M2 x M3
        M2,M3 = self.layers[2].W
        grad_l_wrt_w3 = grad_l_wrt_w3.reshape(M2,M3)

        return grad_l_wrt_w3, grad_l_wrt_y2.T, grad_gamma_3, grad_beta_3
        

    def layer_2_grad(self, grad_l_wrt_w4 ):
        '''Computes and returns the gradient for the 2nd layer.'''
        pass

    def layer_1_grad(self, grad_l_wrt_w4 ):
        '''Computes and returns the gradient for the 1st layer.'''
        pass
        


    
    def grad(self, x, y): 
        '''Returns a gradient for nn4 as a tuple of grad_l_wrt_w1, grad_l_wrt_w2, grad_l_wrt_w3, and grad_l_wrt_w4.'''
        if self.use_batchnorm:
            # set up the size of the dimensions. Storing in one variable not to have to retrieve it multiple times.
            # M_1 = np.shape(self.layers[0].W)[1]
            # M_2 = np.shape(self.layers[1].W)[1]
            # M_3 = np.shape(self.layers[2].W)[1]
            # M_4 = np.shape(self.layers[3].W)[1]
            # K = np.shape(self.layers[0].W)[0]
            # Further set up
            y_hat = self.layers[3].z_tilda

            grad_l_wrt_w4 = self.layer_4_grad(self.layers[3].z,y,y_hat)

            grad_l_wrt_w3 = self.layer_3_grad(grad_l_wrt_w4)








        else:
            raise NotImplementedError("NN4 gradient (backpropagation) Without Batchnorm model not implemented")    

    def emp_loss_grad(self, train_minibatch_X, train_minibatch_y, radnom_W_index):
        '''Calculates the gradient of empirical loss function for a minibatch.'''
        # number of iterations
        N = np.shape(train_minibatch_X)[1]

        # train bminibatch is of size 785 x K, where K is the number of images in the minibatch

        # number of CUDA blocks

        # number of threads
        # total threads = number of grids * number of blocks in each grid * number of threads in each block


        #get the empirical loss image by image
        for tx, ty in zip(train_minibatch_X.T,train_minibatch_y.T):
            emp_loss = self.grad( tx, ty, W)
            sum_img_emp_loss[0] += emp_loss[0]
            sum_img_emp_loss[1] += emp_loss[1]



