# No additional 3rd party external libraries are allowed
import numpy as np

def Adam(model, train_X, train_y):
    #set hyperparameters
    alpha = 0.1
    beta_1 = 0.1
    beta_2 = 0.1
    delta = 0.1
    R = 100
    
    # initially the momentum is zero
    m_r = []   
    w_r = [] 
    s_r = []
    
    m_r_plus_1= []
    s_r_plus_1= [] 
    
    # identify the normalized m_r and s_r
    m_r_plus_1_hat = []
    s_r_plus_1_hat = []
    
    # Set the shapes of the lists
    for i in range(len(model.layers)):
        # Momentum and adaptive step terms for current r rotation 
        m_r.append(np.zeros((np.shape(model.layers[i].W))))
        s_r_plus_1.append(np.zeros((np.shape(model.layers[i].W))))
        
        # Momentum and adaptive step terms for the rotation 1 ahead
        m_r_plus_1.append(np.zeros((np.shape(model.layers[i].W))))
        s_r.append(np.zeros((np.shape(model.layers[i].W))))
        
        # Normalized m_r and s_r
        m_r_plus_1_hat.append(np.zeros((np.shape(model.layers[i].W))))
        s_r_plus_1_hat.append(np.zeros((np.shape(model.layers[i].W))))
        
   
    # Set the shape for w_r
    for layer in model.layers:
        w_r.append(layer.W)
        
    for r in range(1,R+1):         
        # pick a random layer and its index
        random_w_index = np.random.randint(len(model.layers))
        
        # randomly picked W
        random_w_original = model.layers[random_w_index].W
        
        # Reshape the W layer into M x 1 shape so we can easilily pcik j and null out other values
        # and for the reason when W is of shape M x K, we would not be able to do computation
        # for del_L_N_of_w_r_min_1_j for all images at once because matrices would not mach up
        random_w = np.reshape(random_w_original,(np.shape(random_w_original)[0] * np.shape(random_w_original)[1],1))
          
        # identify random initial point j in the vectowr W
        j_index = np.random.randint(len(random_w))
        j = np.asscalar(random_w[j_index])

        #zero out all other parameters in random W other than the jth parameter
        w_r_j = np.zeros(shape=(np.shape(random_w)),dtype=float)
        
        # assign j to that index of j
        w_r_j[j_index] = j
        
        # reshape the one-hot vector to its original shape of random w
        w_r_j = np.reshape(w_r_j,(np.shape(model.layers[random_w_index].W)))
            
        # Get the gradeient of the weights with respect to the one-hot vector w_r_j
        del_L_N_of_w_r = model.emp_loss_grad( train_X, train_y, w_r_j, random_w_index)
        
        w_r_plus_one = []
        
        # For each layer and its gradient perform the update step
        for i in range(len(w_r)):
            # compute momentum term
            m_r_plus_1[i] =  beta_1 * m_r[i] + (1 - beta_1) * del_L_N_of_w_r[i]
    
            # compute the adaptive step (RMS prop)
            s_r_plus_1[i] = beta_2 * s_r[i] + (1 - beta_2) * del_L_N_of_w_r[i] * del_L_N_of_w_r[i]
               
            # normalize the momentum step and the adaptive step
            m_r_plus_1_hat[i] =  m_r_plus_1[i] / (1 - beta_1**r)
            s_r_plus_1_hat[i] =  s_r_plus_1[i] / (1 - beta_2**r)
            
            # Perform the update step
            w_r_plus_one.append(w_r[i] - (alpha / (np.sqrt(s_r_plus_1_hat[i])+np.exp(1))) * m_r_plus_1_hat[i])
            
            
        # update the s_r and m_r
        s_r = s_r_plus_1
        m_r = m_r_plus_1
            
        # update the w_r_j to be the w_r for the next iteration
        w_r = w_r_plus_one

    return w_r_plus_one
    
    