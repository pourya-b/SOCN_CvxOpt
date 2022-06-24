# %% Exercises of "First order methods in optimization by Amir Beck"
# Part 2: ex. 3
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy.linalg import sqrtm
from DPG import DPG
from FDPG import FDPG
# plt.rcParams['text.usetex'] = True

x = np.sin(10*np.arange(1, 41)**3)
y = np.sin(28*np.arange(1, 41)**3)
cl = (2*x < y+0.5)+1
x = np.hstack((x, 0.2))
y = np.hstack((y, -0.2))
cl = np.hstack((cl, 2))
A1 = np.column_stack((x[cl == 1], y[cl == 1]))
A2 = np.column_stack((x[cl == 2], y[cl == 2]))

n = 2
m = 41
x0 = np.zeros((n, 1))
y0 = np.zeros((m, 1))
e = np.ones((m, 1))
zr = np.zeros((m, 1))

# %%
iter_max = 40
A = np.vstack((A1, -1 * A2))
eig_vec, _ = la.eig(A.T @ A)
L = np.max(eig_vec.real)
sigma = 1
Lf = L/sigma


def f(x): return 0.5 * la.norm(x)**2
def g(x): return np.sum(np.max((zr, 1 - A @ x), axis=0))
def x_step(x): return x
def g_prox(x, Lam): return x + Lam * np.max((zr, np.min((e, (1-x)/Lam), axis=0)), axis=0)


F_opt = 8.477981206132888  # by running 1000 iteration of FDPG

dpg_opt = DPG(x_step=x_step, g_prox=g_prox, A=A, Lf=Lf, x0=x0, y0=y0, iter_max=iter_max)
fdpg_opt = FDPG(x_step=x_step, g_prox=g_prox, A=A, Lf=Lf, x0=x0, y0=y0, iter_max=iter_max)

dpg_hist = np.zeros(iter_max)
fdpg_hist = np.zeros(iter_max)

state = dpg_opt
for i, state in enumerate(dpg_opt):
    dpg_hist[i-1] = f(state.x) + g(state.x) - F_opt
sol_DPG = state.x
print(f"solution by DPG: {sol_DPG}")


for i, state in enumerate(fdpg_opt):
    fdpg_hist[i-1] = f(state.x) + g(state.x) - F_opt
sol_FDPG = state.x
print(f"solution by FDPG: {sol_FDPG}")

# %% plotting
X = np.vstack((A1, A2))
x_min = np.min(X[:, 0])
x_max = np.max(X[:, 0])
x_vec = np.arange(x_min, x_max, (x_max-x_min)/20)
y_dpg = -sol_DPG[0]/sol_DPG[1] * x_vec
y_fdpg = -sol_FDPG[0]/sol_FDPG[1] * x_vec

plt.plot(A1[:, 0], A1[:, 1], '*')
plt.plot(A2[:, 0], A2[:, 1], 'd')
plt.plot(x_vec, y_dpg)
plt.plot(x_vec, y_fdpg)
plt.title("different classifiers")
plt.ylim(-2, 2)
plt.legend(("class 1", "class 2", "DPG classifier", "FDPG classifier"))
plt.grid()
# plt.show()

# plt.figure()
# methods_hist = (np.vstack((dpg_hist, fdpg_hist))).T
# plt.plot(methods_hist)
# plt.yscale("log")
# plt.xlabel("iteration")
# plt.ylabel("$F(x^k)-F_{opt}$")
# plt.legend(("DPG", "FDPG"))
# plt.grid()
# plt.show()

plt.savefig("result_images/ex_p2_e3_results.eps")

stop = 1
