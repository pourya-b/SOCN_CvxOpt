# %% proximal gradient method
import numpy as np


class PGM:  # proximal gradient method
    def __init__(self, f_grad, g_prox, Lf, x0, iter_max=1000):
        self.f_grad = f_grad
        self.g_prox = g_prox
        self.Lf = Lf
        self.iter_max = iter_max
        self.x0 = x0

    def __iter__(self):
        self.counter = 0
        self.x = self.x0
        self.gamma = 1/self.Lf
        return self

    def __next__(self):
        if self.counter > self.iter_max:
            print("maximum iteration reached")
            raise StopIteration

        self.x -= self.gamma * self.f_grad(self.x)
        self.x = self.g_prox(self.x)

        self.counter += 1
        return self
