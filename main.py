
import numpy as np
import matplotlib.pylab as plt

# initialization
m, n = 100, 120
a = np.arange(0, m)+1
b = np.arange(0, n)+0.5
A = np.sin(10 * np.outer(a, b)**3)
xi = np.sin(31 * np.arange(1, n+1)**3)
b = A @ xi
lambda1 = 2
lambda2 = 0.5

Iter_max = 100
xk = np.zeros(n)
At = np.transpose(A)
dist = np.zeros(Iter_max)
for i in range(Iter_max):
    tk = 1/np.sqrt(i+1)
    xk = 1/(tk+1) * (xk - tk * At @ np.sign(A @ xk - b))
    dist[i] = np.linalg.norm(xi - xk)

# plt.plot(dist)
# plt.xlabel("iters")
# plt.ylabel("|xk-x*|")
# plt.show()


yk = np.zeros(m)
wk = yk
L = np.linalg.norm(A) ** 2
tk = 1
dist = np.zeros(Iter_max)
def st(x, alpha): return np.maximum(np.abs(x)-alpha, 0) * np.sign(x)


for i in range(Iter_max):
    uk = At @ wk
    yk_ = wk - 1/L * A @ uk + 1/L * st(A@uk - L * wk - b, L) + b
    tk_ = (1+np.sqrt(1+4*tk**2))/2
    wk = yk + ((tk-1)/tk_) * (yk_ - yk)
    yk = yk_
    tk = tk_
    xk = At @ yk
    dist[i] = np.linalg.norm(xi - xk)
