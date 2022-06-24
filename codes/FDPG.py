# %% proximal gradient method
import numpy as np
from copy import deepcopy


class FDPG:  # proximal gradient method
    def __init__(self, x_step, g_prox, A, Lf, x0, y0, iter_max=1000):
        self.x_step = x_step
        self.g_prox = g_prox
        self.Lf = Lf
        self.iter_max = iter_max
        self.x0 = x0
        self.y0 = y0
        self.A = A

    def __iter__(self):
        self.counter = 0
        self.x = self.x0
        self.y = deepcopy(self.y0)
        self.w = deepcopy(self.y0)
        self.gamma = 1/self.Lf
        self.t = 1
        return self

    def __next__(self):
        if self.counter > self.iter_max:
            print("maximum iteration reached")
            raise StopIteration

        self.x = self.x_step(self.A.T @ self.w)
        y_old = deepcopy(self.y)
        self.y = self.w - self.gamma * self.A @ self.x + self.gamma * self.g_prox(self.A @ self.x - self.Lf * self.w, self.Lf)
        t_old = self.t
        self.t = (1+np.sqrt(1+4*self.t**2))/2
        self.w = self.y + ((t_old-1)/self.t) * (self.y - y_old)

        self.counter += 1
        return self
