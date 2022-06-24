# %% Exercises of "First order methods in optimization by Amir Beck"
# Part 2: ex. 3
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.linalg import sqrtm
from ADLPMM import ADLPMM
from CP import CP
# plt.rcParams['text.usetex'] = True

m = 30
n = 25
a = np.arange(m)+1
b = np.arange(n)+0.5
AA = np.sin(10*np.outer(a, b)**3)
xi = np.sin(31*np.arange(1, n+1)**3)
b = AA@xi+np.sin(23*np.arange(1, m+1)**3)+1.5
b = b[:, None]

x0 = np.zeros((n, 1))
y0 = np.zeros((m, 1))
e = np.ones((m, 1))
zr = np.zeros((m, 1))

# %%
iter_max = 60
idx = np.arange(0, n-2)
D1 = np.zeros((n-2, n))
D1[idx, idx] = np.ones(n-2)
D1[idx, idx+1] = -np.ones(n-2)
D2 = np.zeros((n-2, n))
D2[idx, idx+1] = np.ones(n-2)
D2[idx, idx+2] = -np.ones(n-2)

B = -1 * np.vstack((AA, AA, D1, D2))
A = np.eye(B.shape[0])
c = np.zeros((B.shape[0], 1))

eig_vec, _ = la.eig(A.T @ A)
La = np.max(eig_vec.real)
eig_vec, _ = la.eig(B.T @ B)
Lb = np.max(eig_vec.real)

eig_vec, _ = la.eig(AA.T @ AA)
A_root = np.sqrt(np.max(eig_vec.real))


x0 = np.zeros((B.shape[0], 1))
z0 = np.zeros((n, 1))
y0 = np.zeros((B.shape[0], 1))


def ind(x, a):
    q = 0 if all(x >= b) else np.inf
    return q


def h1(x): return -1 * np.sum(np.log(AA @ x - b + 0.00000001)) + np.sum(np.sqrt((D1@x)**2 + (D2@x)**2)) + ind(AA@x, b)
def h2(x): return 0
def h2_prox(x, Lam): return x


def h1_prox(x, Lam):
    z = x[0:m]
    w = x[m:2*m]
    v = x[2*m:2*m+n-2]
    u = x[2*m+n-2:]

    e = np.ones((n-2, 1))

    prox_z = ((z-b) + np.sqrt((z-b)**2+(4*Lam)))/2 + b
    prox_w = np.max((w, b), axis=0)
    Max = np.max((np.sqrt(v**2+u**2), Lam * e), axis=0)
    Comm = 1 - Lam/Max
    prox_v = Comm * v
    prox_u = Comm * u

    prox_x = np.vstack((prox_z, prox_w, prox_v, prox_u))
    return prox_x


F_opt = 105.55646869486733  # by running 10000 iteration of CP

rho_vec = [1, 1/(A_root**2)]

adlpmm_hist = np.zeros((2, iter_max))
cp_hist = np.zeros((1, iter_max))

for k in range(2):
    rho = rho_vec[k]
    alpha = rho * La
    beta = rho * Lb
    adlpmm_opt = ADLPMM(h1_prox, h2_prox, A, B, c, rho, alpha, beta, x0, z0, y0, iter_max)

    state = adlpmm_opt
    for i, state in enumerate(adlpmm_opt):
        adlpmm_hist[k, i-1] = h1(state.z) + h2(state.z) - F_opt
        print(adlpmm_hist[k, i-1])


sigma = 1/(A_root**2)
tau = 1/(A_root**2)
cp_opt = CP(h1_prox, h2_prox, -1*B, tau, sigma, z0, x0, iter_max)

for i, state in enumerate(cp_opt):
    cp_hist[0, i-1] = h1(state.x) + h2(state.x) - F_opt
    print(cp_hist[0, i-1])


plt.figure()
methods_hist = (np.vstack((adlpmm_hist, cp_hist))).T
plt.plot(methods_hist)
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("$F(x^k)-F_{opt}$")
plt.legend(("ADLPMM 1", "ADLPMM 2", "CP"))
plt.grid()
# plt.show()

plt.savefig("ex_p3_e2_1_1_results.eps")

stop = 1
