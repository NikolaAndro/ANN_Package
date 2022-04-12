# No additional 3rd party external libraries are allowed
import numpy as np

def async_sgd(model, train_x_batches, train_y_batches, lr=0.1, R=100):
    '''
    Compute gradient estimate of emp loss on each minibatch in parallel using GPU blocks/threads
    use the most recent gradient estimate computed on a single minibatch to update the weights asynchronously.
    '''
    #TODO
    raise NotImplementedError("Asynchronous minibatch SGD Not Implemented")
