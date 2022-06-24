# %% proximal gradient method
import numpy as np


class DPG:  # proximal gradient method
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
        self.y = self.y0
        self.gamma = 1/self.Lf
        return self

    def __next__(self):
        if self.counter > self.iter_max:
            print("maximum iteration reached")
            raise StopIteration

        self.x = self.x_step(self.A.T @ self.y)
        self.y = self.y - self.gamma * self.A @ self.x + self.gamma * self.g_prox(self.A @ self.x - self.Lf * self.y, self.Lf)

        self.counter += 1
        return self
