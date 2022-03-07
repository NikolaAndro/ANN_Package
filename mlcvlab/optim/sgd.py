# No additional 3rd party external libraries are allowed
from tkinter import W
import numpy as np

#lr = learning rate / step size
#R = Number of iterations / rounds
def SGD(model,train_X, train_y, lr=0.1, R=100):
    '''Computing Random Coordinate Descent (RCD)'''
   
    w_r_minus_1 = []
    for layer in model.layers:
        w_r_minus_1.append(layer.W)
        
    for r in range(R):         
        # pick a random layer and its index
        random_w_index = np.random.randint(len(model.layers))
         
        random_w_original = model.layers[random_w_index].W
        
        #reshape the W layer into M x 1 shape so we can easilily pcik j and null out other values
        # and for the reason when W is of shape M x K, we would not be able to do computation
        # for del_L_N_of_w_r_min_1_j for all images at once because matrices would not mach up
        random_w = np.reshape(random_w_original,(np.shape(random_w_original)[0] * np.shape(random_w_original)[1],1))
          
        # identify random initial point j in the vectowr W
        j_index = np.random.randint(len(random_w))
        j = np.asscalar(random_w[j_index])

        #zero out all other parameters in random W other than the jth parameter
        w_r_min_1_j = np.zeros(shape=(np.shape(random_w)),dtype=float)
        # assign j to that index of j
        w_r_min_1_j[j_index] = j
        
        # reshape the one-hot vector to its original shape of random w
        w_r_min_1_j = np.reshape(w_r_min_1_j,(np.shape(random_w_original)))
            
        # initialize W 0 in the first iteration
        if r == 0:
           w_r_minus_1[random_w_index] = random_w_original
            

        del_L_N_of_w_r_min_1_j = model.emp_loss_grad( train_X, train_y, w_r_min_1_j, random_w_index)
        
        w_r = []
        
        for i in range(len(model.layers)):
            w_r.append(w_r_minus_1[i] - lr * del_L_N_of_w_r_min_1_j[i])
        
        #update the w_r_minus_1_j to be the w_r for the next iteration
        w_r_minus_1 = w_r

    return w_r
    
      