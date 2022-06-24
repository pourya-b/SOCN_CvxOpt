# %% Exercises of "First order methods in optimization by Amir Beck"
# Part 1: ex. 6

import numpy as np


def prox(x, Lambda=1):
    # Amir Beck, prox computations table
    n = np.shape(x.flatten())[0]
    a = np.ones(n)
    b = 3
    l = np.zeros(n)
    u = 2 * np.ones(n)

    P_C = proj_H_box(x/Lambda, a, b, l, u)
    prox = x - Lambda * P_C.flatten()
    return prox


def proj_H_box(x, a, b, l, u, tol=1e-10, gamma=1e-2):
    # Amir Beck, orthogonal projections table
    mu = 0
    x, a, l, u = x[:, None], a[:, None], l[:, None], u[:, None]
    x_b = np.min((np.max((x, l), axis=0), u), axis=0)

    while(np.abs(a.T @ x_b - b) > tol):
        x_mu = x - mu * a
        x_b = np.min((np.max((x_mu, l), axis=0), u), axis=0)
        mu += gamma * (a.T @ x_b - b)

    return x_b


# %% Exercises:

# Part 1: ex. 6:
x = np.array((2, 1, 4, 1, 2, 1))
prox_vec = prox(x, Lambda=1)
print(f"\nThe prox vector is: {prox_vec}\n")
