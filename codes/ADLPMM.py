# %% proximal gradient method
import numpy as np
from copy import deepcopy


class ADLPMM:  # proximal gradient method
    def __init__(self, h1_prox, h2_prox, A, B, c, rho, alpha, beta, x0, z0, y0, iter_max=1000):
        self.h1_prox = h1_prox
        self.h2_prox = h2_prox
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.iter_max = iter_max
        self.x0 = x0
        self.z0 = z0
        self.y0 = y0
        self.A = A
        self.B = B
        self.c = c

    def __iter__(self):
        self.counter = 0
        self.x = deepcopy(self.x0)
        self.y = deepcopy(self.y0)
        self.z = deepcopy(self.z0)

        return self

    def __next__(self):
        if self.counter > self.iter_max:
            print("maximum iteration reached")
            raise StopIteration

        self.x = self.h1_prox(self.x - self.rho/self.alpha * self.A.T @ (self.A @ self.x + self.B @ self.z - self.c + (1/self.rho) * self.y), 1/self.alpha)
        self.z = self.h2_prox(self.z - self.rho/self.beta * self.B.T @ (self.A @ self.x + self.B @ self.z - self.c + (1/self.rho) * self.y), 1/self.beta)
        self.y = self.y + self.rho * (self.A @ self.x + self.B @ self.z - self.c)

        self.counter += 1
        return self
