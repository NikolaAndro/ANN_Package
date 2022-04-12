# No additional 3rd party external libraries are allowed
import numpy as np
from numba import jit

@jit
def async_sgd(model, train_x_batches, train_y_batches, lr=0.1, R=100):
    '''
    Compute gradient estimate of emp loss on each minibatch in parallel using GPU blocks/threads
    use the most recent gradient estimate computed on a single minibatch to update the weights asynchronously.
    '''
    w_r_minus_1 = []
    for layer in model.layers:
        w_r_minus_1.append(layer.W)



