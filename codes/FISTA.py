# %% proximal gradient method
import numpy as np
from copy import deepcopy


class FISTA:  # proximal gradient method
    def __init__(self, f_grad, g_prox, Lf, x0, iter_max=1000):
        self.f_grad = f_grad
        self.g_prox = g_prox
        self.Lf = Lf
        self.iter_max = iter_max
        self.x0 = x0

    def __iter__(self):
        self.counter = 0
        self.t = 1
        self.x = deepcopy(self.x0)
        self.y = deepcopy(self.x0)
        self.gamma = 1/self.Lf
        return self

    def __next__(self):
        if self.counter > self.iter_max:
            print("maximum iteration reached")
            raise StopIteration

        self.x_old = deepcopy(self.x)
        self.y -= self.gamma * self.f_grad(self.y)
        self.x = self.g_prox(self.y)
        t_old = self.t
        self.t = (1 + np.sqrt(1 + 4 * self.t**2))/2
        self.y = self.x + ((t_old - 1)/(self.t)) * (self.x - self.x_old)

        self.counter += 1
        return self
