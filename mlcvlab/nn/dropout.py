# No additional 3rd party external libraries are allowed
import numpy as np

def dropout(z, p, mode='test'):
    '''
    Output : should return a tuple containing 
     - z : z is the output of basis function. Now it will be the output of the dropout
     - p : Dropout param
     - mode : 'test' or 'train'
     - mask : 
        - in train mode, it is the dropout mask
        - in test mode, mask will be None.
    
    sample output: (z=, p=0.5, mode='test',mask=None)
    '''
    # TODO Implement logic for both 'test' and 'train' modes.
    if mode == 'train':
        # Mask for the dropout. List of binomial distribution in size of x.
        # mask = [int(np.random.binomial(n=1, p=p, size=1)) for q in range(len(z))]
        mask = np.random.binomial(1, p, size=z.shape) / p
        # For every distribution == 1 in hte mask, make that element of x = 0.
        # z_drop = [0 if mask[index] == 1 else z[index] for index in range(len(z))]
        z_drop = z * mask

    elif mode == 'test':
        # No mask in testing mode since we want to keep all the nodes active. 
        #What we need to do is to make it matches the training phase expectation, so we scale the layer output with p.
        mask = None
        z_drop = z * p
    
    return z_drop, p, mode, mask

def dropout_grad(z, mask, mode='train'):
    '''Gradient of the dropout.
    Parameters:
    - del_L_wrt_y: the gradient of loss function implemented in losses.py.

    Output:  del_L_wrt_z_tilda
    '''
    e = 0.001

    if mode == 'train':
        # grad_L_wrt_z_tilda = 1 / mask + e
        grad_L_wrt_z_tilda = z * mask
    
    return grad_L_wrt_z_tilda
    # elif mode == 'test':
    #     raise NotImplementedError("Gradiant of Dropout - Test Not Implemented")