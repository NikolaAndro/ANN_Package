# No additional 3rd party external libraries are allowed
import numpy as np

#lr = learning rate / step size
#R = Number of iterations / rounds
def SGD(model,train_X, train_y, lr=0.1, R=100):
    '''Computing Random Coordinate Descent (RCD)'''
    updated_weights = None

    # identify a random vector W
    random_w_index = np.randint(0,len(train_X))
    w_r_minus_1 = train_X[random_w_index]

    for r in range(R):
        
        # identify random initial point j in the vectowr W
        j_index = np.randint(0, len(w_r_minus_1))
        j = w_r_minus_1[j_index]

        #zero out all other parameters in w_r_min_1 other than the jth parameter
        w_r_min_1_j = np.zeros(shape=(1,train_X.shape[1]))
        w_r_min_1_j[j_index] = j

        # Compute the gradient of empirical loss with tesprct to w_r_min_1_j
        # This is an estimate of del_L_n_hat
        del_L_N_of_w_r_min_1_j = model.emp_loss_grad( train_X, train_y, w_r_min_1_j, w_r_minus_1)

        # compute the update step for any model
        w_r = w_r_minus_1 - lr * (del_L_N_of_w_r_min_1_j)
        
        #update the w_r_minus_1_j to be the w_r for the next iteration
        # w_r_minus_1 = w_r

        #update the final gradient
        # Adding bunch of one-hot vectors
        np.append(updated_weights,w_r)


    # to get the full gradient we need to sum up all one-hot vectors
    #full_gradient = np.sum(updated_weights)
    #return full_gradient

    return updated_weights

    #return Updated_Weights
    
      