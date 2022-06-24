# %% Exercises of "First order methods in optimization by Amir Beck"
# Part 1: ex. 9
import numpy as np


def prox_t(x, Lambda=1, tol=1e-10, gamma=1e-2):
    # Amir Beck, prox computations table
    t = x
    if (x.flatten()).shape[0] > 1:
        raise Warning("In prox, the argument should be scalar")
        return 0

    def inner_fun(w): return w ** 3 - x * (w ** 2) - Lambda
    while(np.abs(inner_fun(t)) > tol):
        t -= gamma * (inner_fun(t))

    return t


def prox(X, Lambda=1):
    Gamma, U_T = np.linalg.eig(X)
    n = X.shape[0]

    inner_prox_vec = np.zeros(n)
    for i in range(n):
        inner_prox_vec[i] = prox_t(Gamma[i], Lambda=Lambda)

    prox_vec = U_T @ np.diag(inner_prox_vec) @ U_T.T
    return prox_vec


# %% Exercises:

# Part 1: ex. 6:
X = np.array(((3, 1), (1, 4)))
prox_mat = prox(X, Lambda=1)
print(f"\nThe prox matrix is: {prox_mat}\n")
