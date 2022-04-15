# No additional 3rd party external libraries are allowed
import numpy as np

def async_sgd(model, x_train_batches, y_train_batches,eps, lr=0.1, R=100, gamma = 0.99, beta = 0.5):
    '''
    Compute gradient estimate of emp loss on each minibatch in parallel using GPU blocks/threads
    use the most recent gradient estimate computed on a single minibatch to update the weights asynchronously.
    '''
    # setup
    w_r_minus_1 = np.array([])
    for index,layer in enumerate(model.layers):
        w_r_minus_1[index] = layer.W
    gamma_beta_minus_1 = np.array([gamma, beta])

    #Run epochs
    for r in range(R):
        #iterate over minibatches
        for x_minibatch, y_minibatch in zip(x_train_batches, y_train_batches):

            if model.use_batchnorm:
                del_L_N_of_w_r_min_1, del_L_N_of_gamma_and_beta_min_1 = model.emp_loss_grad(x_minibatch, y_minibatch, eps)
                # Update the gamma and beta
                gamma_beta = gamma_beta_minus_1 - lr * del_L_N_of_gamma_and_beta_min_1
            else:
                del_L_N_of_w_r_min_1 = model.emp_loss_grad(x_minibatch, y_minibatch, eps)

            # Update the weights
            w_r = w_r_minus_1 - lr * del_L_N_of_w_r_min_1

            w_r_minus_1 = w_r
    
    if model.use_batchnorm:
        return [w_r, gamma_beta]
    else:
        return w_r

            




