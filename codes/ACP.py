# %% proximal gradient method
import numpy as np
from copy import deepcopy


class ACP:  # proximal gradient method
    def __init__(self, f_prox, g_prox, A, gamma, tau0, sigma0, x0, y0, iter_max=1000):
        self.f_prox = f_prox
        self.g_prox = g_prox
        self.tau = tau0
        self.sigma = sigma0
        self.iter_max = iter_max
        self.x0 = x0
        self.y0 = y0
        self.A = A
        self.gamma = gamma

    def __iter__(self):
        self.counter = 0
        self.x = deepcopy(self.x0)
        self.y = deepcopy(self.y0)
        self.x_old = deepcopy(self.x0)
        self.theta = 1/np.sqrt(1+self.gamma * self.tau)

        return self

    def __next__(self):
        if self.counter > self.iter_max:
            print("maximum iteration reached")
            raise StopIteration

        temp = self.y + self.sigma * self.A @ (self.x + self.theta * (self.x - self.x_old))
        self.y = temp - self.sigma * self.f_prox(temp/self.sigma, 1/self.sigma)

        self.x_old = deepcopy(self.x)
        self.x = self.g_prox(self.x - self.tau * self.A.T @ self.y, self.tau)

        self.theta = 1/np.sqrt(1+self.gamma * self.tau)
        self.tau *= self.theta
        self.sigma /= self.theta

        self.counter += 1
        return self
