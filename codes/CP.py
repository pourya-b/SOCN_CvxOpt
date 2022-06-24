# %% proximal gradient method
import numpy as np
from copy import deepcopy


class CP:  # proximal gradient method
    def __init__(self, f_prox, g_prox, A, tau, sigma, x0, y0, iter_max=1000):
        self.f_prox = f_prox
        self.g_prox = g_prox
        self.tau = tau
        self.sigma = sigma
        self.iter_max = iter_max
        self.x0 = x0
        self.y0 = y0
        self.A = A

    def __iter__(self):
        self.counter = 0
        self.x = deepcopy(self.x0)
        self.y = deepcopy(self.y0)

        return self

    def __next__(self):
        if self.counter > self.iter_max:
            print("maximum iteration reached")
            raise StopIteration

        x_old = deepcopy(self.x)
        self.x = self.g_prox(self.x - self.tau * self.A.T @ self.y, self.tau)
        temp = self.y + self.sigma * self.A @ (2*self.x - x_old)
        self.y = temp - self.sigma * self.f_prox(temp/self.sigma, 1/self.sigma)

        self.counter += 1
        return self
