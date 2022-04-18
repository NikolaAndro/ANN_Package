# No additional 3rd party external libraries are allowed
import numpy as np

def dropout(b, p, mode='test'):
    '''
    Output : should return a tuple containing 
     - b : b is the output of batch norm layern. dim: M x 1
     - p : Dropout param
     - mode : 'test' or 'train'
     - mask : 
        - in train mode, it is the dropout mask
        - in test mode, mask will be None.
    
    sample output: (b_drop=, p=0.5, mode='test',mask=None)
    '''
    # TODO Implement logic for both 'test' and 'train' modes.
    if mode == 'train':
        # Mask for the dropout. List of binomial distribution in size of x.
        # A simpler and commonly used alternative called Inverted Dropout scales 
        # the output activation during training phase by 1p so that we can leave 
        # the network during testing phase untouched.
        mask = np.random.binomial(1, p, size=b.shape)  # dim: M x 1
        
        # For every distribution == 1 in hte mask, make that element of x = 0.
        # When the output of the neuron is scaled to 0, it does not contribute 
        # any further during both forward and backward pass, which is essentially dropout.
        b_drop = b * mask  # dim: M x 1

    elif mode == 'test':
        # No mask in testing mode since we want to keep all the nodes active. 
        # What we need to do is to make it matches the training phase expectation, so we scale the layer output with p.
        mask = None
        b_drop = np.dot((1-p),b) # dim: M x 1
        # Dividing with p in training part. Hence, b stays untouched in testing phase
        #b_drop = b
        
    
    return b_drop, p, mode, mask

def dropout_grad(z, mask, mode='train'):
    '''Gradient of the dropout. Returning diagonalized gradient.'''
 
    if mode == 'train':
        #grad_w_wrt_b = np.dot(mask  * np.identity(np.shape(mask)[0]), np.identity(np.shape(z)[0]) * z)
        grad_w_wrt_b = mask  * np.identity(np.shape(mask)[0])
    else:
        raise ValueError("dropout_grad() can only be called in train mode.")
    return grad_w_wrt_b
   