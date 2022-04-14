from math import gamma
from torch import batch_norm


class Layer():
    def __init__(self, W, activation, z, z_tilda, batch_norm, gamma, beta, dropout):
        self.W  = W
        self.activation = activation
        self.z = z
        self.z_tilda = z_tilda
        self.batch_norm = batch_norm #tuple (b, cache) and cache is a tuple (x_normalized, x_mean, x_variance, gamma)
        self.gamma = gamma
        self.beta = beta
        self.y_out = dropout # tuple (b_drop, p, mode, mask)
        