import numpy as np

class AdamOptimizer():

    def __init__(self, dim: int, d: int): # dim * d
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.m = np.zeros([dim, d])
        self.v = np.zeros([dim, d])

        self.iter = 0
    

    def get_update(self, grad, learning_rate):
        self.iter += 1

        lr_alpha = learning_rate*np.sqrt(1.0-self.beta2**self.iter)/(1.0-self.beta1**self.iter)

        self.m += (1-self.beta1)*(grad-self.m)
        self.v += (1-self.beta2)*(grad**2-self.v)

        return lr_alpha * self.m / (np.sqrt(self.v) + 1e-7)
