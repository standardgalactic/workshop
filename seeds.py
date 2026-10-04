import numpy as np
from scipy.stats import norm
exec(open("sim.py").read().split("N = 200000")[0])
N=200000
def w1(a,b): return np.mean(np.abs(np.sort(a)-np.sort(b)))
def euler(K,x0):
    x=x0.copy(); d=1.0/K
    for i in range(K):
        x=x+d*v(x,i*d)
    return x
# ---- repeated seeds: uncertainty for last-step ratios and the W1 floor ----
def last_step_ratio(D, seed, n=200000):
    r = np.random.default_rng(seed); s = 1 - D
    k = r.integers(0, 2, n); x1 = MU[k] + SIG * r.standard_normal(n)
    xs = s * x1 + (1 - s) * r.standard_normal(n)
    y = mean_post(xs, s); pos = y[y > 0]; neg = y[y <= 0]
    return 0.5 * (pos.std() + neg.std()) / SIG
for D in [1/8, 1/16, 1/32, 1/64, 1/256]:
    vals = np.array([last_step_ratio(D, 100 + i) for i in range(6)])
    s = 1 - D; pred = s * SIG / np.sqrt(tau2(s))
    print("D", D, "mean", vals.mean().round(4), "sd", vals.std(ddof=1).round(4), "pred", pred.round(4))
w = []
for i in range(6):
    r = np.random.default_rng(500 + i)
    z0 = r.standard_normal(N)
    y = euler(2048, z0)
    k = r.integers(0, 2, N); t = MU[k] + SIG * r.standard_normal(N)
    w.append(w1(y, t))
print("W1 K=2048 over seeds mean", np.mean(w), "sd", np.std(w, ddof=1), "min", min(w), "max", max(w))
exact = SIG*np.sqrt(2/np.pi)*np.exp(-A**2/(2*SIG**2)) + A*(1-2*norm.cdf(-A/SIG))
print("E|x1| exact", exact)
print("s_c exact", 1/(1+np.sqrt(A**2-SIG**2)))
