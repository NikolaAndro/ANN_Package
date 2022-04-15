from html2text import element_style
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
            z_1 = np.dot(self.layers[0].W.T , x) # dim: M1 x K . K x 1 => M1 x 1
            self.layers[0].z = z_1

            z_1_tilda = relu(z_1) # dim: M1 x 1
            self.layers[0].z_tilda = z_1_tilda
            # apply batchnorm
            # batch_cache = self.layers[0].batch_norm[2]
            b_1 = batchnorm(z_1_tilda, gamma=0.1, beta=0.1, eps=0.001, mode=curent_mode )
            self.layers[0].batch_norm = b_1
            self.layers[0].gamma = b_1[1][3]
            self.layers[0].beta = b_1[1][4]
            # apply dropout; B_1[0] dim: M1 x 1
            y_1 = dropout(b_1[0], p = 0.5, mode = curent_mode) # results in tuple (b_drop, p, mode, mask) 
            # y_1_out -> dim: M1 x 1
            self.layers[0].y_out = y_1[0]

            #************************** LAYER 2 ************************
            z_2 = np.dot(self.layers[1].W.T , y_1)  # dim: M2 x M1 . M1 x 1 => M1 x 1
            self.layers[1].z  = z_2

            z_2_tilda = relu(z_2)
            self.layers[1].z_tilda = z_2_tilda
            # apply batchnorm
            b_2 = batchnorm(z_2_tilda, gamma=0.1, beta=0.1, eps=0.001, mode=curent_mode )
            self.layers[1].batch_norm = b_2
            self.layers[1].gamma = b_2[1][3]
            self.layers[1].beta = b_2[1][4]
            # apply dropout
            y_2 = dropout(b_2[0], p = 0.5, mode = curent_mode)
            self.layers[1].y_out = y_2[0]

            #************************** LAYER 3 ************************
            z_3 = np.dot(self.layers[2].W.T , y_2) # dim: M3 x M2 . M2 x 1 => M3 x 1
            self.layers[2].z = z_3

            z_3_tilda = relu(z_3)
            self.layers[2].z_tilda = z_3_tilda
            # apply batchnorm
            b_3 = batchnorm(z_3_tilda, gamma=0.1, beta=0.1, eps=0.001, mode=curent_mode )
            self.layers[2].batch_norm = b_3
            self.layers[2].gamma = b_2[2][3]
            self.layers[2].beta = b_2[2][4]
            # apply dropout
            y_3 = dropout(b_3[0], p = 0.5, mode = curent_mode)
            self.layers[2].y_out = y_3[0]

            #************************** LAYER 4 ************************
            z_4 = np.dot(self.layers[3].W.T , y_3) # dim: M4 x M3 . M3 x 1 => M4 x 1
            self.layers[3].z = z_4
            
            y_hat = sigmoid(z_4)
            self.layers[3].y_out = y_hat # just to keep the value in memory for the grad funciton

        else:
            #************************** LAYER 1 ************************
            z_1 = np.dot(self.layers[0].W.T , x)
            self.layers[0].z = z_1

            z_1_tilda = relu(z_1)
            self.layers[0].z_tilda = z_1_tilda
            # apply dropout
            y_1 = dropout(z_1_tilda, p = 0.5, mode = curent_mode)
            self.layers[0].y_out = y_1[0]

            #************************** LAYER 2 ************************
            z_2 = np.dot(self.layers[1].W.T , y_1)
            self.layers[1].z  = z_2

            z_2_tilda = relu(z_2)
            self.layers[1].z_tilda = z_2_tilda
            # apply dropout
            y_2 = dropout(z_2_tilda, p = 0.5, mode = curent_mode)
            self.layers[1].y_out = y_2[0]

            #************************** LAYER 3 ************************
            z_3 = np.dot(self.layers[2].W.T , y_2)
            self.layers[2].z = z_3

            z_3_tilda = relu(np.dot(self.layers[2].W , z_3))
            self.layers[2].z_tilda = z_3_tilda
            # apply dropout
            y_3 = dropout(z_3_tilda, p = 0.5, mode = curent_mode)
            self.layers[2].y_out = y_3[0]

            #************************** LAYER 4 ************************
            z_4 = np.dot(self.layers[3].W.T , y_3)
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
        '''Computes and returns the gradient for the 3rd layer.
        parameters:
        - grad_l_wrt_y3 - dim: M4 x M3 
        '''
        y3 = self.layers[2].y_out # dim: M3 x 1
        mask_3 = self.layers[2].y_out[3] # result of dropout layer is tuple (z_drop, p, mode, mask)
        grad_y3_wrt_b3 = dropout_grad(y3,mask_3) # M3 x M3 diagonal matrix of mask
        # grad_l_wrt_y3 is already transposed when returned from layer_4_grad()
        # grad_l_wrt_y3 => 1 x M3
        # grad_y3_wrt_b3 => M3 x M3
        grad_l_wrt_b3 = np.dot(grad_l_wrt_y3, grad_y3_wrt_b3) # dim: 1 x M3

        #******** updates for batch norm gamma and beta *******
        grad_b3_wrt_z3_tilda, grad_gamma_3, grad_beta_3 = batchnorm_grad(grad_l_wrt_b3.T,self.layers[2].z_tilda, eps, self.layers[2].batch_norm[1])
        # diagonalize grad_b3_wrt_z3_tilda
        grad_b3_wrt_z3_tilda = grad_b3_wrt_z3_tilda * np.identity(np.shape(grad_b3_wrt_z3_tilda)[0]) #dim :M3 x M3
        grad_l_wrt_z3_tilda = np.dot(grad_l_wrt_b3,grad_b3_wrt_z3_tilda) # dim: 1 x M3


        grad_z3_tilda_wrt_z3 =  relu_grad(self.layers[2].z)   # M3 x 1
        grad_z3_tilda_wrt_z3 = grad_z3_tilda_wrt_z3 *  np.identity(np.shape(grad_z3_tilda_wrt_z3)[0])# diagonalize to amke it M3 x M3
        grad_l_wrt_z3 = np.dot(grad_l_wrt_z3_tilda,grad_z3_tilda_wrt_z3) # 1 x M3 . M3 x M3 => 1 x M3


        grad_z3_wrt_y2  = self.layers[2].W.T  # dim: M3 x M2
        grad_l_wrt_y2 =  np.dot(grad_l_wrt_z3,grad_z3_wrt_y2) # dim: 1 x M3 . M3 x M2 => 1 x M2 
        
        # *************** Weights *************
        grad_z3_wrt_w3 = [] # dim: M3 x M2 x M3
        for z in range(self.layers[2].z):
            grad_z3_wrt_w3.append(np.dot(z, self.layers[2].W)) # W3-> M2xM3 ; z = 1x1

        grad_l_wrt_w3 = np.dot(grad_l_wrt_z3,grad_z3_wrt_w3)    # dim: 1 x M3 .  M3 x M2 x M3=> 1 x M2 x M3
        # reshape the dimentsions from 1 x M2 x M3 to just M2 x M3
        M2,M3 = self.layers[2].W
        grad_l_wrt_w3 = grad_l_wrt_w3.reshape(M2,M3)

        return grad_l_wrt_w3, grad_l_wrt_y2, grad_gamma_3, grad_beta_3
        

    def layer_2_grad(self, grad_l_wrt_y2, eps):
        '''Computes and returns the gradient for the 2nd layer.
        Parameters:
        - grad_l_wrt_y2 : dim: 1 x M2 (no need for transpose)
        '''
        y2 = self.layers[1].y_out #dim: M2 x 1
        mask_2 = self.layers[1].y_out[3] # result of dropout layer is tuple (z_drop, p, mode, mask) dim: M2 x 1
        grad_y2_wrt_b2 = dropout_grad(y2,mask_2) # M2 x M2 diagonal matrix of mask
        # grad_l_wrt_y2 is already transposed when returned from layer_3_grad()
        # grad_l_wrt_y2 => 1 x M2
        # grad_y2_wrt_b2 => M2 x M2
        grad_l_wrt_b2 = np.dot(grad_l_wrt_y2, grad_y2_wrt_b2) # dim: 1 x M2

        #******** updates for batch norm gamma and beta *******
        grad_b2_wrt_z2_tilda, grad_gamma_2, grad_beta_2 = batchnorm_grad(grad_l_wrt_b2.T,self.layers[1].z_tilda, eps, self.layers[1].batch_norm[1]) 
        # diagonalize grad_b2_wrt_z2_tilda
        grad_b2_wrt_z2_tilda = grad_b2_wrt_z2_tilda * np.identity(np.shape(grad_b2_wrt_z2_tilda)[0]) # dim: M2 x M2
        grad_l_wrt_z2_tilda = np.dot(grad_l_wrt_b2,grad_b2_wrt_z2_tilda) # dim: 1 x M2 . 

        grad_z2_tilda_wrt_z2 =  relu_grad(self.layers[1].z)   # M2 x 1
        grad_z2_tilda_wrt_z2 = grad_z2_tilda_wrt_z2 *  np.identity(np.shape(grad_z2_tilda_wrt_z2)[0])# diagonalize to amke it M2 x M2
        grad_l_wrt_z2 = np.dot(grad_l_wrt_z2_tilda,grad_z2_tilda_wrt_z2) # 1 x M2 . M2 x M2 => 1 x M2

        grad_z2_wrt_y1  = self.layers[1].W.T  # dim: M2 x M1
        grad_l_wrt_y1 = np.dot(grad_l_wrt_z2, grad_z2_wrt_y1) # dim: 1 x M2 . M2 x M1 => 1 x M1 

        # *************** Weights *************
        grad_z2_wrt_w2 = [] # dim: M2 x M1 x M2
        for z in range(self.layers[1].z):
            grad_z2_wrt_w2.append(np.dot(z, self.layers[1].W)) # W2-> M1xM2 ; z = 1x1

        grad_l_wrt_w2 = np.dot(grad_l_wrt_z2,grad_z2_wrt_w2)    # dim: 1 x M2 .  M2 x M1 x M2=> 1 x M1 x M2
        # reshape the dimentsions from 1 x M1 x M2 to just M1 x M2
        M1,M2 = np.shape(self.layers[1].W)
        grad_l_wrt_w2 = grad_l_wrt_w2.reshape(M1,M2)

        return grad_l_wrt_w2, grad_l_wrt_y1, grad_gamma_2, grad_beta_2

    def layer_n_grad(self, layer_number, batch_n, eps, **kwargs):
        '''Computes and returns the gradient for the n-th layer, where n can be 1, 2, or 3.
        Parameters:
        - grad_l_wrt_yn : dim: 1 x M (no need for transpose)
        - layer_number : starts at 
        - eps : epsilon
        - z_4 : z from layer 4
        - y : image label
        - y_hat : prediction
        Output for layer 4:
        - grad_l_wrt_w, grad_l_wrt_y.T
        Output for any other layer"
        - grad_l_wrt_w, grad_l_wrt_y, grad_gamma, grad_beta
        '''
        if layer_number in range(1,4):
            if batch_n:
                drop_input = self.layers[layer_number - 1].batch_norm[0] #dim: M x 1
            else:
                drop_input = self.layers[layer_number - 1].z_tilda #dim: M x 1

            mask = self.layers[layer_number - 1].y_out[3] # result of dropout layer is tuple (z_drop, p, mode, mask) dim: M x 1
            
            #naming it this way since it can be b if there is batch norm and z_tilda if there is no batch norm layer
            grad_y_wrt_b_or_z_tilda = dropout_grad(drop_input,mask) # M x M diagonal matrix of mask
            
            if batch_n:
                # grad_l_wrt_y is already transposed when returned from layer_n_grad()
                # grad_l_wrt_y => 1 x M
                # grad_y_wrt_b => M x M
                grad_l_wrt_b = np.dot(kwargs['grad_l_wrt_yn'], grad_y_wrt_b_or_z_tilda) # dim: 1 x M

                #******** updates for batch norm gamma and beta *******
                grad_b_wrt_z_tilda, grad_gamma, grad_beta = batchnorm_grad(grad_l_wrt_b.T,self.layers[layer_number - 1].z_tilda, eps, self.layers[layer_number - 1].batch_norm[1]) 
                # diagonalize grad_b_wrt_z_tilda
                grad_b_wrt_z_tilda = grad_b_wrt_z_tilda * np.identity(np.shape(grad_b_wrt_z_tilda)[0]) # dim: M x M
                grad_l_wrt_z_tilda = np.dot(grad_l_wrt_b,grad_b_wrt_z_tilda) # dim: 1 x M . 
            else:
                grad_l_wrt_z_tilda = np.dot(kwargs['grad_l_wrt_yn'], grad_y_wrt_b_or_z_tilda) # dim: 1 x M

            # This part same for with and without batch norm layer
            grad_z_tilda_wrt_z =  relu_grad(self.layers[layer_number - 1].z)   # M x 1
            grad_z_tilda_wrt_z = grad_z_tilda_wrt_z *  np.identity(np.shape(grad_z_tilda_wrt_z)[0])# diagonalize to amke it M x M
            grad_l_wrt_z = np.dot(grad_l_wrt_z_tilda,grad_z_tilda_wrt_z) # 1 x M . M x M => 1 x M


            # When we reach the layer 1, we do not need this computation
            if layer_number != 1:
                grad_z_wrt_y  = self.layers[layer_number - 1].W.T  # dim: M x Mx
                grad_l_wrt_y = np.dot(grad_l_wrt_z, grad_z_wrt_y) # dim: 1 x M . M x Mx => 1 x Mx 

            # *************** Weights *************
            grad_z_wrt_w = [] # dim: M x Mx x M
            for z in range(self.layers[layer_number - 1].z):
                grad_z_wrt_w.append(np.dot(z, self.layers[layer_number - 1].W)) # W-> Mx x M ; z = 1x1

            grad_l_wrt_w = np.dot(grad_l_wrt_z,grad_z_wrt_w)    # dim: 1 x M .  M x Mx x M=> 1 x Mx x M
            # reshape the dimentsions from 1 x Mx x M to just Mx x M
            m_x,m = np.shape(self.layers[layer_number - 1].W)
            grad_l_wrt_w = grad_l_wrt_w.reshape(m_x,m)

            # Dimensions:
            # grad_l_wrt_w - Mx x M
            # grad_l_wrt_y - 1 x Mx
            # grad_gamma, grad_beta - scalars
            if batch_n:
                if layer_number != 1:
                    return [grad_l_wrt_w, grad_l_wrt_y, grad_gamma, grad_beta]
                else:
                    return [grad_l_wrt_w, grad_gamma, grad_beta]
            else:
                if layer_number != 1:
                    return [grad_l_wrt_w, grad_l_wrt_y]
                else:
                    return grad_l_wrt_w
        else:
            raise ValueError("Layer number can be in range [1,3] in this NN architecture.")
        
    

    
    def grad(self, x, y, eps): 
        '''Returns a gradient for nn4 as a tuple of grad_l_wrt_w1, grad_l_wrt_w2, grad_l_wrt_w3, and grad_l_wrt_w4.
        and it will also calculate the change in gamma and beta for every layer.
        Parameters:
        - x : training x (1 image)
        - y : training y (1 image label)
        - eps : epsilon used in computations'''
        if self.use_batchnorm:
            y_hat = self.layers[3].z_tilda

            grad_l_wrt_w4, grad_l_wrt_y3_transpose = self.layer_4_grad(self.layers[3].z,y,y_hat)

            grad_l_wrt_w3, grad_l_wrt_y2_transpose, grad_gamma_3, grad_beta_3 = self.layer_n_grad(3, self.use_batchnorm, eps,grad_l_wrt_yn = grad_l_wrt_y3_transpose)

            grad_l_wrt_w2, grad_l_wrt_y1_transpose, grad_gamma_2, grad_beta_2 = self.layer_n_grad(2, self.use_batchnorm, eps,grad_l_wrt_yn = grad_l_wrt_y2_transpose)

            grad_l_wrt_w1, grad_gamma_1, grad_beta_1 = self.layer_n_grad(1, self.use_batchnorm, eps,grad_l_wrt_yn = grad_l_wrt_y1_transpose)

            # Collect the gradient of the weights for each  layer as well as gamma and beta for each layer
            weights_grad = np.array[grad_l_wrt_w1, grad_l_wrt_w2, grad_l_wrt_w3, grad_l_wrt_w4]
            gamma_beta_grad = np.array[[grad_gamma_1, grad_beta_1],[grad_gamma_2, grad_beta_2],[grad_gamma_3, grad_beta_3]]

            return weights_grad, gamma_beta_grad

        else:
            y_hat = self.layers[3].z_tilda

            grad_l_wrt_w4, grad_l_wrt_y3_transpose = self.layer_4_grad(self.layers[3].z,y,y_hat)

            grad_l_wrt_w3, grad_l_wrt_y2_transpose = self.layer_n_grad(3, self.use_batchnorm, eps,grad_l_wrt_yn = grad_l_wrt_y3_transpose)

            grad_l_wrt_w2, grad_l_wrt_y1_transpose = self.layer_n_grad(2, self.use_batchnorm, eps,grad_l_wrt_yn = grad_l_wrt_y2_transpose)

            grad_l_wrt_w1 = self.layer_n_grad(1, self.use_batchnorm, eps,grad_l_wrt_yn = grad_l_wrt_y1_transpose)

            # Collect the gradient of the weights for each  layer
            weights_grad = np.array[grad_l_wrt_w1, grad_l_wrt_w2, grad_l_wrt_w3, grad_l_wrt_w4]

            return weights_grad

    def emp_loss_grad(self, train_minibatch_x, train_minibatch_y, eps):
        '''Calculates the gradient of empirical loss function for a minibatch.
        - train_minibatch_x: images in the minibatch
        - train_minibatch_y: labels for the images in the images
        - eps: epsilon
        '''
        # number of iterations
        N = np.shape(train_minibatch_x)[1]

        sum_weights_emp_loss = np.array([np.zeros((np.shape(self.layers[0].W))),\
                np.zeros((np.shape(self.layers[1].W))), \
                    np.zeros((np.shape(self.layers[2].W))), \
                        np.zeros((np.shape(self.layers[3].W))),])

        sum_gamma_beta = np.array([[0,0],[0,0],[0,0]])

        # train bminibatch is of size 785 x K, where K is the number of images in the minibatch          
        #iterate over images and labels in each batch
        for image, label in zip(train_minibatch_x, train_minibatch_y):


            if self.use_batchnorm:
                emp_loss_weights, emp_loss_gamma_beta = self.grad(image, label, eps)
                sum_gamma_beta += emp_loss_gamma_beta
            else:
                emp_loss_weights = self.grad(image, label, eps)

            #add weights, gammas and betas
            sum_weights_emp_loss += emp_loss_weights
            

        emp_loss_weights = (1 / N) * sum_weights_emp_loss
        
        if self.use_batchnorm:
            emp_loss_gamma_beta = (1/N) * sum_gamma_beta
            return [emp_loss_weights,emp_loss_gamma_beta]
        else:
            return emp_loss_weights


