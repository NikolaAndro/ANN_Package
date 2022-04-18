
# No additional 3rd party external libraries are allowed
import numpy as np
# tqdm - package used to shoe a progress bar when loops executing
# tqdm - "progress" in arabic and obriviation for  "Te Quiero DeMaciado" 
# in Spanish (I love you so much)
from tqdm import tqdm 

from numba import cuda
from mlcvlab_GPU.models import nn4

def sync_sgd(model, x_train_batches, y_train_batches, lr=0.1, R=100):
    '''
    Compute gradient estimate of emp loss on each mini batch in-parallel using GPU blocks/threads.
    Wait for all results and aggregate results by calling cuda.synchronize(). For more details, refer to https://thedatafrog.com/en/articles/cuda-kernel-python
    Compute update step synchronously
    '''

    # send the minibatches to GPU so we reduce the transfer data time for each minibatch call
    x_train_batches_gpu = cuda.to_device(x_train_batches)
    y_train_batches_gpu = cuda.to_device(y_train_batches)
    
    # send the model to gpu as well since it contains all variables needed for computations
    model_gpu = cuda.to_device(model)
    
    # setup
    w_r_minus_1 = []
    gamma_beta_minus_1 = []
    # gpu output array
    weights_gpu = []
    gamma_beta_gpu = []
    
    for index, layer in enumerate(model.layers):
        w_r_minus_1.append(layer.W)
        weights_gpu.append(layer.W)
        if index != 3:
            gamma_beta_minus_1.append([layer.gamma, layer.beta])
            gamma_beta_gpu.append([0,0])

    w_r_minus_1 = np.array(w_r_minus_1)
    gamma_beta_minus_1 = np.array(gamma_beta_minus_1)
    
    weights_gpu = np.array(weights_gpu)
    gamma_beta_gpu = np.array(gamma_beta_gpu)

    # Parallely compute the empirical loss of every minibatch accross different blocks
    # On my machine, there is NVIDIA TITAN RTX GPU, which has 7.5 CC (Compute Capability)
    
    # we then want to use K number of blocks, where K is the number of minibatches
    blocks_per_grid = np.shape(x_train_batches)[2]
    threads_per_block = 60000 // blocks_per_grid
    
    # The max num of threads per block on this card is 1024
    if threads_per_block > 1000:
        raise ValueError("The sumber of minibatches must be >= 60.")
        
        
    #Run epochs
    for r in tqdm(range(R)):
        if model.use_batchnorm:
            
            # creating the array for the kernel function that will be populated with the results from gpu
            emp_loss_weights_gamma_beta = [weights_gpu, gamma_beta_gpu]
            
            # send the output to  gpu
            emp_loss_weights_gamma_beta_gpu = cuda.to_device(emp_loss_weights_gamma_beta)
            
            # parallely execue the emp loss on each minibatch
            model_gpu.emp_loss_grad[blocks_per_grid,threads_per_block](x_train_batches_gpu,y_train_batches_gpu,emp_loss_weights_gamma_beta_gpu)
            
            # syncronize every thread
            cuda.syncthreads()
            
            # copy the results back to CPU
            emp_loss_weights_gamma_beta = emp_loss_weights_gamma_beta_gpu.copy_to_host()
            
            del_L_N_of_w_r_min_1, del_L_N_of_gamma_and_beta_min_1 = emp_loss_weights_gamma_beta
            
            # Update the gamma and beta
            gamma_beta = gamma_beta_minus_1 - lr * del_L_N_of_gamma_and_beta_min_1
            
        else:
            # creating the array for the kernel function that will be populated with the results from gpu
            emp_loss_weights = [weights_gpu]
            
            # send the output to gpu
            emp_loss_weights_gpu = cuda.to_device(emp_loss_weights)
            
            # parallely execue the emp loss on each minibatch
            model.emp_loss_grad[blocks_per_grid,threads_per_block](x_train_batches_gpu,y_train_batches_gpu, emp_loss_weights_gpu)
            
            # syncronize every thread
            cuda.syncthreads()
            
            # copy the results back to CPU
            emp_loss_weights = emp_loss_weights_gpu.copy_to_host()
            
            del_L_N_of_w_r_min_1 = emp_loss_weights_gamma_beta

        # Update the weights
        w_r = w_r_minus_1 - lr * del_L_N_of_w_r_min_1

        w_r_minus_1 = w_r
    
    # copy the model back to cpu
    model = model_gpu.copy_to_host()
    
    for i in range(4):
        model.layers[i].W = w_r[i]
        if model.use_batchnorm and i != 3:
            model.layers[i].gamma = gamma_beta[i][0]
            model.layers[i].beta = gamma_beta[i][1]

    return model