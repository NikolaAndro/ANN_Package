# No additional 3rd party external libraries are allowed
import numpy as np

def async_sgd(model, x_train_batches, y_train_batches,eps, lr=0.1, R=100):
    '''
    Compute gradient estimate of emp loss on each minibatch in parallel using GPU blocks/threads
    use the most recent gradient estimate computed on a single minibatch to update the weights asynchronously.
    '''
    w_r_minus_1 = []
    for layer in model.layers:
        w_r_minus_1.append(layer.W)
    
    for r in range(R):
        #iterate over minibatches
        for x_minibatch, y_minibatch in zip(x_train_batches, y_train_batches):

            # del_L_N_of_w_r_min_1 = model.emp_loss_grad(x_minibatch, y_minibatch, eps)

            # w_r = w_r_minus_1 - lr * del_L_N_of_w_r_min_1

            # w_r_minus_1 = w_r
            pass




