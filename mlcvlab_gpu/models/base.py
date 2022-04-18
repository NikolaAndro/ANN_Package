from math import gamma


class Layer():
    def __init__(self, W, activation, z, z_tilda, batch_norm, gamma, beta, dropout_output = None, dropout_param = 0.5):
        self.W  = W
        self.activation = activation
        self.z = z
        self.z_tilda = z_tilda
        self.batch_norm = batch_norm #tuple (b, cache) and cache is a tuple (z_tilda_normalized, z_tilda_mean, z_tilda_variance, gamma, beta, running_mean, running_variance)
        self.gamma = gamma
        self.beta = beta
        self.y_out = dropout_output # tuple (b_drop, p, mode, mask)
        self.dropout_param = dropout_param
        