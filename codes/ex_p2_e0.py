# %% Exercises of "First order methods in optimization by Amir Beck"
# Part 2: ex. 0
import numpy as np
import matplotlib.pyplot as plt
from PGM import PGM
from FISTA import FISTA
from VFISTA import VFISTA
# plt.rcParams['text.usetex'] = True

m, n = 100, 120
lambda1, lambda2 = 2, 0.5
a = np.arange(0, m)+1
b = np.arange(0, n)+0.5
A = np.sin(10 * np.outer(a, b)**3)
xi = np.sin(31 * np.arange(1, n+1)**3)
b = A @ xi
b = b[:, None]

x0 = np.zeros((n, 1))
e = np.ones((n, 1))
zr = np.zeros((n, 1))

# %%
iter_max = 100
eig_vec, _ = np.linalg.eig(A.T @ A)
Lf = np.max(eig_vec.real) + lambda1
sigma = np.min(eig_vec.real) + lambda1
kappa = Lf/sigma


def f(x): return 0.5 * np.linalg.norm(A @ x - b) ** 2 + (lambda1/2) * np.linalg.norm(x) ** 2
def g(x): return lambda2 * np.linalg.norm(x, 1)
def f_grad(x): return A.T @ (A @ x - b) + lambda1 * x
def g_prox(x): return np.max((zr, np.abs(x) - (lambda2/Lf) * e), axis=0) * np.sign(x)


F_opt = 73.8213461807307  # by running 2000 iteration of PGM

pgm_opt = PGM(f_grad, g_prox, Lf, x0, iter_max=iter_max)
fista_opt = FISTA(f_grad, g_prox, Lf, x0, iter_max=iter_max)
vfista_opt_1 = VFISTA(f_grad, g_prox, Lf, kappa, x0, iter_max=iter_max)

pgm_hist = np.zeros(iter_max)
fista_hist = np.zeros(iter_max)
vfista_hist_1 = np.zeros(iter_max)
vfista_hist_2 = np.zeros(iter_max)

state = pgm_opt
for i, state in enumerate(pgm_opt):
    pgm_hist[i-1] = f(state.x) + g(state.x) - F_opt
print(f"first 4 componenets of PGM: {state.x[0:4]}")

for i, state in enumerate(fista_opt):
    fista_hist[i-1] = f(state.x) + g(state.x) - F_opt
print(f"first 4 componenets of FISTA: {state.x[0:4]}")

for i, state in enumerate(vfista_opt_1):
    vfista_hist_1[i-1] = f(state.x) + g(state.x) - F_opt
print(f"first 4 componenets of V-FISTA 1: {state.x[0:4]}")


# %%
Lf = np.max(eig_vec.real)
sigma = lambda1
kappa = Lf/sigma


def f_grad_2(x): return A.T @ (A @ x - b)
def g_prox_2(x): return np.max((zr, np.abs(x/(lambda1/Lf + 1)) - (lambda2/(Lf+lambda1)) * e), axis=0) * np.sign(x/(lambda1/Lf + 1))


vfista_opt_2 = VFISTA(f_grad_2, g_prox_2, Lf, kappa, x0, iter_max=iter_max)
for i, state in enumerate(vfista_opt_2):
    vfista_hist_2[i-1] = f(state.x) + g(state.x) - F_opt
print(f"first 4 componenets of V-FISTA 2: {state.x[0:4]}")

methods_hist = (np.vstack((pgm_hist, fista_hist, vfista_hist_1, vfista_hist_2))).T
plt.plot(methods_hist)
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("$F(x^k) - F_{opt}$")
plt.legend(("PGM", "FISTA", "V-FISTA 1", "V-FISTA 2"))
plt.grid()
# plt.show()

plt.savefig("ex_p2_e0_results.eps")

stop = 1
