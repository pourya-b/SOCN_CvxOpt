# %% Exercises of "First order methods in optimization by Amir Beck"
# Part 2: ex. 1
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.linalg import sqrtm
from PGM import PGM
from FISTA import FISTA
# plt.rcParams['text.usetex'] = True

n = 30
w = np.arange(1, n**2+1)
www = w.reshape((n, n)).T
A = np.sin(93*www**3)
Q = A.T @ A
b = 10*np.sin(27*np.arange(1, n+1)**3)
c = b @ la.solve(Q, b)+1
DD = np.sin(15*www**3)
Z = sqrtm(la.inv(DD@DD.T))
D = Z @ DD
b = b[:, None]

x0 = np.ones((n, 1))
e = np.ones((n, 1))
zr = np.zeros((n, 1))
# %%
iter_max = 1001

eig_vec, _ = la.eig(Q)
Lf = np.max(eig_vec.real)/np.sqrt(c - b.T @ la.inv(Q) @ b)


def f(x): return np.sqrt(x.T @ Q @ x + 2 * b.T @ x + c)
def g(x): return 0.2 * la.norm(D @ x, 1)
def f_grad(x): return (2*b + 2 * Q @ x)/(2*f(x))


def g_prox(x):
    y = (np.max((zr, np.abs(D @ x) - (0.2/Lf) * e), axis=0) * np.sign(D @ x))
    return x + D.T @ (y - D @ x)


F_opt = 25.947266223377028  # by running 10000 iteration of FISTA

pgm_opt = PGM(f_grad, g_prox, Lf, x0, iter_max=iter_max)
fista_opt = FISTA(f_grad, g_prox, Lf, x0, iter_max=iter_max)

pgm_hist = np.zeros(iter_max)
fista_hist = np.zeros(iter_max)

state = pgm_opt
for i, state in enumerate(pgm_opt):
    pgm_hist[i-1] = f(state.x) + g(state.x) - F_opt
    if np.mod(i-1, 100) == 0:
        print(f"iter\;{i}:\; F(x^k)-F_opt: {pgm_hist[i-1]}\\")

for i, state in enumerate(fista_opt):
    fista_hist[i-1] = f(state.x) + g(state.x) - F_opt
    if np.mod(i-1, 100) == 0:
        print(f"iter\;{i}:\; F(x^k)-F_opt: {fista_hist[i-1]}\\")

methods_hist = (np.vstack((pgm_hist, fista_hist))).T
plt.plot(methods_hist)
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("$F(x^k)-F_{opt}$")
plt.legend(("PGM", "FISTA"))
plt.grid()
# plt.show()

plt.savefig("result_images/ex_p2_e1_results.eps")

stop = 1
