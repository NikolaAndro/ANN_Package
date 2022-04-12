# No additional 3rd party external libraries are allowed
import numpy as np

def sync_sgd(model, X_train_batches, y_train_batches, lr=0.1, R=100):
    '''
    Compute gradient estimate of emp loss on each mini batch in-parallel using GPU blocks/threads.
    Wait for all results and aggregate results by calling cuda.synchronize(). For more details, refer to https://thedatafrog.com/en/articles/cuda-kernel-python
    Compute update step synchronously
    '''
    #TODO
    raise NotImplementedError("Synchronous minibatch SGD Not Implemented")
