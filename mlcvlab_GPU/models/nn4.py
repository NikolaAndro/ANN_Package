# f rom html2text import element_style
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
        # randomize the initial gamma and beta
        gamma, beta = np.random.uniform(0.01,1), np.random.uniform(0.01,1)
        eps = 0.000001
        self.layers = [
            Layer(None, relu, None, None, (None,(None,None,None,None,None,1.,1.)) , gamma = gamma, beta=beta, dropout_output = None, dropout_param=0.8),
            Layer(None, relu, None, None, (None,(None,None,None,None,None,1.,1.)), gamma = gamma, beta=beta, dropout_output = None,dropout_param=0.5),
            Layer(None, relu, None, None, (None,(None,None,None,None,None,1.,1.)), gamma = gamma, beta=beta, dropout_output = None,dropout_param=0.5),
            Layer(None, sigmoid, None, None, (None,(None,None,None,None,None,1.,1.)), gamma = gamma, beta=beta, dropout_output = None)]
        
        self.use_batchnorm = use_batchnorm

        #used in dropout implementation
        self.dropout_param = dropout_param

        self.epsilon = eps
    
    # Make this a device (GPU) function since it will only be used on GPU
    @cuda.jit(device=True)
    def nn4(self, x, curent_mode):
        '''Returns the prediction of the neural network based on input iamge x.\
            In pytorch this is model.forward().
            Parameters:
            - x: input image
            - current_mode: "train" or "test"
            '''
        if self.use_batchnorm:
            #************************** LAYER 1 ************************
            z_1 = np.dot(self.layers[0].W.T , x) # dim: M1 x K . K x 1 => M1 x 1
            self.layers[0].z = z_1

            z_1_tilda = relu(z_1) # dim: M1 x 1
            self.layers[0].z_tilda = z_1_tilda
            # apply batchnorm
            # batch_cache = self.layers[0].batch_norm[2]
            b_1 = batchnorm(z_1_tilda, gamma=self.layers[0].gamma, beta=self.layers[0].beta, eps=self.epsilon,\
                            running_mean = self.layers[0].batch_norm[1][5], running_variance = self.layers[0].batch_norm[1][6], mode=curent_mode )
            if curent_mode != 'test':
                self.layers[0].batch_norm = b_1
                self.layers[0].gamma = b_1[1][3]
                self.layers[0].beta = b_1[1][4]
                
            # apply dropout; B_1[0] dim: M1 x 1
            y_1 = dropout(b_1[0], p = self.layers[0].dropout_param, mode = curent_mode) # results in tuple (b_drop, p, mode, mask) 
            # y_1_out -> dim: M1 x 1
            self.layers[0].y_out = y_1

            #************************** LAYER 2 ************************
            z_2 = np.dot(self.layers[1].W.T , y_1[0])  # dim: M2 x M1 . M1 x 1 => M1 x 1
            self.layers[1].z  = z_2

            z_2_tilda = relu(z_2)
            self.layers[1].z_tilda = z_2_tilda
            # apply batchnorm
            b_2 = batchnorm( z_2_tilda, gamma=self.layers[1].gamma, beta=self.layers[1].beta, eps=self.epsilon,\
                            running_mean = self.layers[1].batch_norm[1][5], running_variance = self.layers[1].batch_norm[1][6], mode=curent_mode )
            if curent_mode != 'test':
                self.layers[1].batch_norm = b_2
                self.layers[1].gamma = b_2[1][3]
                self.layers[1].beta = b_2[1][4]
            # apply dropout
            y_2 = dropout(b_2[0], p = self.layers[1].dropout_param, mode = curent_mode)
            self.layers[1].y_out = y_2

            #************************** LAYER 3 ************************
            z_3 = np.dot(self.layers[2].W.T , y_2[0]) # dim: M3 x M2 . M2 x 1 => M3 x 1
            self.layers[2].z = z_3

            z_3_tilda = relu(z_3)
            self.layers[2].z_tilda = z_3_tilda
            # apply batchnorm

            b_3 = batchnorm(z_3_tilda, gamma=self.layers[2].gamma, beta=self.layers[2].beta, eps=self.epsilon,\
                            running_mean = self.layers[2].batch_norm[1][5], running_variance = self.layers[2].batch_norm[1][6], mode=curent_mode )
            if curent_mode != 'test':
                self.layers[2].batch_norm = b_3
                self.layers[2].gamma = b_2[1][3]
                self.layers[2].beta = b_2[1][4]
            # apply dropout
            y_3 = dropout(b_3[0], p = self.layers[2].dropout_param, mode = curent_mode)
            self.layers[2].y_out = y_3

            #************************** LAYER 4 ************************
            y_3_out = y_3[0]
            y_3_out = np.reshape(y_3[0], (np.shape(y_3_out)[0],1))
            z_4 = np.dot(self.layers[3].W.T , y_3_out) # dim: M4 x M3 . M3 x 1 => M4 x 1
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
            y_1 = dropout(z_1_tilda, p = self.layers[0].dropout_param, mode = curent_mode)
            self.layers[0].y_out = y_1

            #************************** LAYER 2 ************************
            z_2 = np.dot(self.layers[1].W.T , y_1[0])
            self.layers[1].z  = z_2

            z_2_tilda = relu(z_2)
            self.layers[1].z_tilda = z_2_tilda
            # apply dropout
            y_2 = dropout(z_2_tilda, p = self.layers[1].dropout_param, mode = curent_mode)
            self.layers[1].y_out = y_2

            #************************** LAYER 3 ************************
            z_3 = np.dot(self.layers[2].W.T , y_2[0])
            self.layers[2].z = z_3

            z_3_tilda = relu(z_3)
            self.layers[2].z_tilda = z_3_tilda
            # apply dropout
            y_3 = dropout(z_3_tilda, p = self.layers[2].dropout_param, mode = curent_mode)
            self.layers[2].y_out = y_3

            #************************** LAYER 4 ************************
            y_3_out = y_3[0]
            y_3_out = np.reshape(y_3[0], (np.shape(y_3_out)[0],1))
            z_4 = np.dot(self.layers[3].W.T , y_3[0])
            self.layers[3].z = z_4
            y_hat = sigmoid(z_4)
            self.layers[3].y_out = y_hat # just to keep the value in memory for the grad funciton
        
        # Since this is binary classification problem of predicting if a digit is odd or even,
        # we will use a treshold to devide results
        if y_hat >= 0.5:
            y_hat = 1
        else:
            y_hat = 0
            
        return y_hat

    def get_grid_dim(self, minibatch_x, minibatch_y):
        '''Returns the size of the CUDA grid assuming that each block is 16 x 32 size. '''
        return int(np.ceil([minibatch_x/16])[0]), int(np.ceil([minibatch_y/32])[0])
    
    # Make this a device (GPU) function since it will only be used on GPU
    @cuda.jit(device=True)
    def layer_4_grad(self,z_4,y,y_hat):
        '''Computes and returns the gradient for the 4th (last) layer.'''
        grad_y_hat_wrt_z4 = sigmoid_grad(z_4)
        grad_l_wrt_y_hat = l2_grad(y,y_hat)
        grad_l_wrt_z4 = np.dot(grad_l_wrt_y_hat, grad_y_hat_wrt_z4) # 1 x 1

        grad_z4_wrt_y3 = self.layers[3].W # returning M3 x M4 matrix = 80 x 1
        grad_l_wrt_y3 = grad_l_wrt_z4 * grad_z4_wrt_y3 # M3 x M4 = 80 x 1

        grad_z4_wrt_w4 = self.layers[2].y_out[0] # y3 - dim: M3 x M4 => M3 x 1 since M4 = 1
        grad_l_wrt_w4 = grad_l_wrt_z4 * grad_z4_wrt_w4

        # Returning grad_l_wrt_y3 transpose just so we reduce the computation time
        # for every next call of grad_l_wrt_y3 that should be transpose.
        # e.g. in layer_3_grad, layer_2_grad , layer_1_grad
        return grad_l_wrt_w4.T, grad_l_wrt_y3.T

    @cuda.jit
    def layer_n_grad(self, layer_number, batch_n, grad_l_wrt_yn):
        '''Computes and returns the gradient for the n-th layer, where n can be 1, 2, or 3.
        Parameters:
        - layer_number : in range [1-3]
        - batch_n : check if using batchnorm
        - grad_l_wrt_yn : dim: 1 x M (no need for transpose)
        
        Output if with batchnorm:
        - grad_l_wrt_w, grad_l_wrt_y, grad_gamma, grad_beta
        
        Output if without batchnorm:
        - grad_l_wrt_w, grad_l_wrt_y
        '''
        
        # get the 
        if layer_number in range(1,4):
            # if batch_n:
            #     drop_input = self.layers[layer_number - 1].batch_norm[0] #dim: M x 1
            # else:
            #     drop_input = self.layers[layer_number - 1].z_tilda #dim: M x 1

            mask = self.layers[layer_number - 1].y_out[3] # result of dropout layer is tuple (z_drop, p, mode, mask) dim: M x 1
            drop_output = self.layers[layer_number -1].y_out[0]
            #naming it this way since it can be b if there is batch norm and z_tilda if there is no batch norm layer
            grad_y_wrt_b_or_z_tilda = dropout_grad(drop_output,mask) # M x M diagonal matrix of mask
            
            if batch_n:
                # grad_l_wrt_y is already transposed when returned from layer_n_grad()
                # grad_l_wrt_y => 1 x M
                # grad_y_wrt_b => M x M
                #grad_l_wrt_yn = grad_l_wrt_yn
                grad_l_wrt_b = np.dot(grad_l_wrt_yn, grad_y_wrt_b_or_z_tilda) # dim: 1 x M

                #******** updates for batch norm gamma and beta *******
                grad_b_wrt_z_tilda, grad_gamma, grad_beta = batchnorm_grad(grad_l_wrt_b.T,self.layers[layer_number - 1].z_tilda,eps=self.epsilon, \
                                                                           cache = self.layers[layer_number - 1].batch_norm[1]) 
                # diagonalize grad_b_wrt_z_tilda
                grad_b_wrt_z_tilda = grad_b_wrt_z_tilda * np.identity(np.shape(grad_b_wrt_z_tilda)[0]) # dim: M x M
                grad_l_wrt_z_tilda = np.dot(grad_l_wrt_b,grad_b_wrt_z_tilda) # dim: 1 x M . 
            else:
                grad_l_wrt_z_tilda = np.dot(grad_l_wrt_yn, grad_y_wrt_b_or_z_tilda) # dim: 1 x M

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
            for z_elem in self.layers[layer_number - 1].z:
                grad_z_wrt_w.append(np.dot(z_elem, self.layers[layer_number - 1].W)) # W-> Mx x M ; z = 1x1
            grad_z_wrt_w = np.array(grad_z_wrt_w)
            
            # ISSUE:  The issue is that np.dot(a,b) for multidimensional arrays makes the dot product of the 
            # last dimension of a with the second-to-last dimension of b
            grad_z_wrt_w = np.reshape(grad_z_wrt_w,(np.shape(grad_z_wrt_w)[1],np.shape(grad_z_wrt_w)[0],np.shape(grad_z_wrt_w)[2])) # dim: Mx x M x M
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
        
    

    # Make this a device (GPU) function since it will only be used on GPU
    @cuda.jit(device=True)
    def grad(self, x, y): 
        '''Returns a gradient for nn4 as a tuple of grad_l_wrt_w1, grad_l_wrt_w2, grad_l_wrt_w3, and grad_l_wrt_w4.
        and it will also calculate the change in gamma and beta for every layer. In pytorch this is model.forward().
        Parameters:
        - x : training x (1 image)
        - y : training y (1 image label)
        - eps : epsilon used in computations'''
        if self.use_batchnorm:

            y_hat = self.layers[3].y_out

            grad_l_wrt_w4, grad_l_wrt_y3_transpose = self.layer_4_grad(self.layers[3].z,y,y_hat)

            grad_l_wrt_w3, grad_l_wrt_y2_transpose, grad_gamma_3, grad_beta_3 = self.layer_n_grad(3, self.use_batchnorm,grad_l_wrt_yn = grad_l_wrt_y3_transpose)

            grad_l_wrt_w2, grad_l_wrt_y1_transpose, grad_gamma_2, grad_beta_2 = self.layer_n_grad(2, self.use_batchnorm,grad_l_wrt_yn = grad_l_wrt_y2_transpose)

            grad_l_wrt_w1, grad_gamma_1, grad_beta_1 = self.layer_n_grad(1, self.use_batchnorm,grad_l_wrt_yn = grad_l_wrt_y1_transpose)

            # Collect the gradient of the weights for each  layer as well as gamma and beta for each layer
            weights_grad = np.array([grad_l_wrt_w1, grad_l_wrt_w2, grad_l_wrt_w3, grad_l_wrt_w4])
            gamma_beta_grad = np.array([[grad_gamma_1, grad_beta_1],[grad_gamma_2, grad_beta_2],[grad_gamma_3, grad_beta_3]])

            return weights_grad, gamma_beta_grad

        else:
            y_hat = self.layers[3].y_out

            grad_l_wrt_w4, grad_l_wrt_y3_transpose = self.layer_4_grad(self.layers[3].z,y,y_hat)

            grad_l_wrt_w3, grad_l_wrt_y2_transpose = self.layer_n_grad(3, self.use_batchnorm,grad_l_wrt_yn = grad_l_wrt_y3_transpose)

            grad_l_wrt_w2, grad_l_wrt_y1_transpose = self.layer_n_grad(2, self.use_batchnorm,grad_l_wrt_yn = grad_l_wrt_y2_transpose)

            grad_l_wrt_w1 = self.layer_n_grad(1, self.use_batchnorm,grad_l_wrt_yn = grad_l_wrt_y1_transpose)

            # shape is (M3,M4) and it results in (80,). Need to make it (80,1) to perform operations on it later.
            grad_l_wrt_w4 = np.reshape(grad_l_wrt_w4, (np.shape(grad_l_wrt_w4)[0],1))
            # Collect the gradient of the weights for each  layer
            weights_grad = np.array([grad_l_wrt_w1, grad_l_wrt_w2, grad_l_wrt_w3, grad_l_wrt_w4])

            return weights_grad
    
    # This is going to be the kernel for the GPU that is going to find the empirical loss of every minibatch in paralel
    # Lists of minibatches are already sent to GPU in the setup
    # kernels cannot explicitly return a value; all result data must be written to an array passed to the function. Hence, we
    # pass emp_loss_weights_gamma_beta array
    @cuda.jit
    def emp_loss_grad(self,x_train_batches_gpu , y_train_batches_gpu, emp_loss_weights_gamma_beta):
        '''Calculates the gradient of empirical loss function for a minibatch.
        - x_train_batches_gpu - input array of x minibatches that needs to be paralelized
        - y_train_batches_gpu - input array of y minibatches that needs to be paralelized
        - emp_loss_weights_gamma_beta: lista that contains change in weights for all 4 layers to be updated after running the changes on gpu as
        well as change in gamma nad beta if batcnorm is used. Structure: [emp_loss_weights, emp_loss_gamma_beta ]
        '''
        
        # get the unique index for the current thread in the whole grid
        idx = cuda.grid(1)
        
        # number of iterations
        N = np.shape(x_train_batches_gpu[0])[1]

        sum_weights_emp_loss = np.array([np.zeros((np.shape(self.layers[0].W))),\
                np.zeros((np.shape(self.layers[1].W))), \
                    np.zeros((np.shape(self.layers[2].W))), \
                        np.zeros((np.shape(self.layers[3].W))),])

        sum_gamma_beta = np.array([[0.,0.],[0.,0.],[0.,0.]])

        # train bminibatch is of size 785 x K, where K is the number of images in the minibatch          
        #iterate over images and labels in each batch
        for image, label in zip(x_train_batches_gpu.T, y_train_batches_gpu.T):


            if self.use_batchnorm:
                # forward pass
                _ = self.nn4(image,curent_mode='train')
                # backward pass
                emp_loss_weights, emp_loss_gamma_beta = self.grad(image, label)
                
                emp_loss_gamma_beta = np.reshape(emp_loss_gamma_beta,(np.shape(emp_loss_gamma_beta)[0],np.shape(emp_loss_gamma_beta)[1]))
                # used to update gamma and beta
                sum_gamma_beta += emp_loss_gamma_beta
            else:
                # forward pass
                _ = self.nn4(image,curent_mode='train')
                # backward pass
                emp_loss_weights = self.grad(image, label)

            #add weights, gammas and betas
            sum_weights_emp_loss += emp_loss_weights
            
        # average the weights
        emp_loss_weights = (1 / N) * sum_weights_emp_loss
        
        
        # update the output
        if self.use_batchnorm:
            emp_loss_weights_gamma_beta = [emp_loss_weights, emp_loss_gamma_beta]
        else:
            emp_loss_weights_gamma_beta = [emp_loss_weights]
        
        
        # if self.use_batchnorm:
        #     #average gamma and beta
        #     emp_loss_gamma_beta = (1/N) * sum_gamma_beta
        #     return [emp_loss_weights,emp_loss_gamma_beta]
        # else:
        #     return emp_loss_weights


