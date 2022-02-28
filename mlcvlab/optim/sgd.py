# No additional 3rd party external libraries are allowed
import numpy as np

#lr = learning rate / step size
#R = Number of iterations / rounds
def SGD(model,train_X, train_y, lr=0.1, R=100):
    
    updated_weights = None

    for r in range(R):
        # identify a random vector W
        random_w_index = np.randint(0,len(train_X))
        random_w = train_X[random_w_index]

        # identify random initial point j in the vectowr W
        j_index = np.randint(0, len(random_w))
        j = random_w[j_index]

        #zero out all other parameters in w_r_min_1 other than the jth parameter
        w_r_min_1_j = np.zeros(shape=(1,train_X.shape[1]))
        w_r_min_1_j[j_index] = j

        # Compute the gradient of empirical loss with tesprct to w_r_min_1_j
        del_L_N_of_w_r_min_1_j = model.emp_loss_grad( train_X, train_y, w_r_min_1_j, layer)

        # compute the update step for any model
        w_r = w_r_min_1_j - lr * (del_L_N_of_w_r_min_1_j)

        #update the final gradient
        np.append(updated_weights,w_r)

    return updated_weights

    #return Updated_Weights
    
      