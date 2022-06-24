# %% Exercises of "First order methods in optimization by Amir Beck"
# Part 2: ex. 3
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.linalg import sqrtm
from ACP import ACP
from FDPG import FDPG
# plt.rcParams['text.usetex'] = True

m = 30
n = 25
a = np.arange(m)+1
b = np.arange(n)+0.5
AA = np.sin(10*np.outer(a, b)**3)
xi = np.sin(31*np.arange(1, n+1)**3)
b = AA@xi+np.sin(23*np.arange(1, m+1)**3)+1.5
b = b[:, None]

# %%
iter_max = 2500
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
rho = 1/(A_root)  # adjustable
alpha = rho * La
beta = rho * Lb
tau = 1/(A_root**2)
sigma = 1/(A_root**2)
Lf = Lb/1

x0 = np.zeros((B.shape[0], 1))
z0 = np.zeros((n, 1))
y0 = np.zeros((B.shape[0], 1))


def ind(x, a):
    q = 0 if all(x >= b) else np.inf
    return q


def h1(x): return -1 * np.sum(np.log(AA @ x - b + 0.00000001)) + np.sum(np.sqrt((D1@x)**2 + (D2@x)**2)) + ind(AA@x, b)
def h2(x): return 0.5 * la.norm(x) ** 2
def h2_prox(x, Lam): return x/(1+Lam)
def h2_step(x): return x


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


F_opt = 178.56934879989103  # by running 10000 iteration of ACP

acp_opt = ACP(h1_prox, h2_prox, -1*B, 1, tau, sigma, z0, x0, iter_max)
fdpg_opt = FDPG(h2_step, h1_prox, -1*B, Lf, z0, x0, iter_max)

acp_hist = np.zeros(iter_max)
fdpg_hist = np.zeros(iter_max)

for i, state in enumerate(acp_opt):
    acp_hist[i-1] = h1(state.x) + h2(state.x)-F_opt
    print(acp_hist[i-1])

for i, state in enumerate(fdpg_opt):
    fdpg_hist[i-1] = h1(state.x) + h2(state.x)-F_opt
    print(fdpg_hist[i-1])

plt.figure()
methods_hist = (np.vstack((acp_hist, fdpg_hist))).T
# methods_hist = acp_hist
plt.plot(methods_hist)
plt.yscale("log")
plt.xlabel("iteration")
plt.ylabel("$F(x^k)-F_{opt}$")
plt.legend(("ACP", "FDPG"))
plt.grid()
# plt.show()

plt.savefig("ex_p3_e2_2_1_results.eps")

stop = 1
