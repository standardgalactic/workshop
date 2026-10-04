import numpy as np, json, os
os.makedirs("/tmp/work3", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
A, SIG = 1.0, 0.10          # modes at +-A, within-mode std SIG
MU = np.array([-A, A]); PI = np.array([0.5, 0.5])

def tau2(s):
    return s**2 * SIG**2 + (1 - s)**2

def post(x, s):
    """posterior weights and component posterior means of x1 given x_s=x"""
    t2 = tau2(s)
    logw = np.log(PI)[None, :] - 0.5 * (x[:, None] - s * MU[None, :])**2 / t2
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw); w /= w.sum(axis=1, keepdims=True)
    mk = MU[None, :] + (s * SIG**2 / t2) * (x[:, None] - s * MU[None, :])
    return w, mk

def mean_post(x, s):
    w, mk = post(x, s)
    return (w * mk).sum(axis=1)

def var_post(x, s):
    w, mk = post(x, s)
    m = (w * mk).sum(axis=1)
    ck = SIG**2 * (1 - s)**2 / tau2(s)
    return (w * (ck + (mk - m[:, None])**2)).sum(axis=1)

def v(x, s):
    return (mean_post(x, s) - x) / (1 - s)

def target(n):
    k = rng.integers(0, 2, n)
    return MU[k] + SIG * rng.standard_normal(n)

def w1(a, b):
    return np.mean(np.abs(np.sort(a) - np.sort(b)))

N = 200000
x0 = rng.standard_normal(N)
ref = target(N)

# Euler with K steps, final step ends at s=1 (exact field at s<1 only evaluated at s=(K-1)/K)
def euler(K, x0):
    x = x0.copy(); d = 1.0 / K
    for i in range(K):
        s = i * d
        x = x + d * v(x, s)
    return x

rows = []
for K in [1, 2, 3, 4, 8, 16, 32, 64, 256, 2048]:
    y = euler(K, x0)
    gap = np.mean(np.abs(y) < A / 2)
    # within-mode std: samples assigned by sign
    pos = y[y > 0]; neg = y[y <= 0]
    wstd = 0.5 * (pos.std() + neg.std())
    rows.append(dict(K=K, gap=float(gap), wstd=float(wstd),
                     frac_pos=float((y > 0).mean()),
                     W1=float(w1(y, ref)),
                     mean_abs_mode=float(0.5 * (np.abs(pos).mean() + np.abs(neg).mean()))))
for r in rows:
    print(r)

# oracle last step: true x_s at s=1-D, then output m(x_s,s)
orc = []
for D in [0.5, 0.25, 0.125, 1/16, 1/32, 1/64, 1/256]:
    s = 1 - D
    k = rng.integers(0, 2, N)
    x1 = MU[k] + SIG * rng.standard_normal(N)
    xs = s * x1 + (1 - s) * rng.standard_normal(N)
    y = mean_post(xs, s)
    pos = y[y > 0]; neg = y[y <= 0]
    wstd = 0.5 * (pos.std() + neg.std())
    pred = np.sqrt(s**2 * SIG**4 / tau2(s))      # within-component std of m_k part (w~1)
    gap = np.mean(np.abs(y) < A / 2)
    orc.append(dict(D=D, wstd=float(wstd), pred=float(pred), ratio=float(wstd / SIG),
                    pred_ratio=float(pred / SIG), gap=float(gap), W1=float(w1(y, ref))))
for r in orc:
    print(r)

# commitment: posterior weight of correct mode along true noisy paths
S = np.linspace(0.02, 0.98, 49)
ent = []; lossfloor = []; jac_max = []
for s in S:
    k = rng.integers(0, 2, 40000)
    x1 = MU[k] + SIG * rng.standard_normal(40000)
    xs = s * x1 + (1 - s) * rng.standard_normal(40000)
    w, _ = post(xs, s)
    p = np.clip(w[:, 1], 1e-12, 1 - 1e-12)
    ent.append(float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p)))))
    lossfloor.append(float(np.mean(var_post(xs, s)) / (1 - s)**2))
    h = 1e-4
    xg = np.linspace(-1.5, 1.5, 3001)
    J = (v(xg + h, s) - v(xg - h, s)) / (2 * h)
    jac_max.append(float(J.max()))
ent = np.array(ent); lossfloor = np.array(lossfloor); jac_max = np.array(jac_max)

# time of half-committed: first s where mean posterior entropy < 0.5*log2
half = float(S[np.argmax(ent < 0.5 * np.log(2))])
print("half-commit s", half, " predicted (1-s)=s*A ->", 1 / (1 + A))
print("jac max peak at s=", float(S[np.argmax(jac_max)]), "value", float(jac_max.max()))

def clean(o):
    if isinstance(o,float) and o!=o: return None
    if isinstance(o,dict): return {k:clean(v) for k,v in o.items()}
    if isinstance(o,list): return [clean(v) for v in o]
    return o
json.dump(clean(dict(rows=rows, orc=orc, half=half, S=S.tolist(), ent=ent.tolist(),
               loss=lossfloor.tolist(), jac=jac_max.tolist())),
          open("/tmp/work3/results.json", "w"), indent=1)

# ---------- figures ----------
plt.rcParams.update({"font.size": 9, "font.family": "serif"})

# Fig 1: trajectories and marginals
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.4, 1]))
xt = rng.standard_normal(60)
T = 400; ts = np.linspace(0, 1 - 1e-3, T)
traj = np.zeros((T, 60)); x = xt.copy(); traj[0] = x
for i in range(1, T):
    d = ts[i] - ts[i - 1]
    x = x + d * v(x, ts[i - 1]); traj[i] = x
for j in range(60):
    c = "tab:red" if traj[-1, j] > 0 else "tab:blue"
    ax[0].plot(ts, traj[:, j], color=c, lw=0.6, alpha=0.8)
ax[0].axvline(1 / (1 + A), color="k", ls=":", lw=0.8)
ax[0].text(1 / (1 + A) + 0.01, -2.3, r"$s_c$", fontsize=9)
ax[0].set_xlabel("$s$"); ax[0].set_ylabel("$x_s$"); ax[0].set_title("Exact-field trajectories", fontsize=9)
ax[0].set_ylim(-2.6, 2.6)
xx = np.linspace(-2.6, 2.6, 400)
for s, ls in [(0.0, "-"), (0.3, "--"), (0.5, "-."), (0.8, ":"), (1.0, "-")]:
    if s < 1:
        yv = np.zeros(400)
        for k in range(2):
            t2 = tau2(s)
            yv += 0.5 * np.exp(-0.5 * (xx - s * MU[k])**2 / t2) / np.sqrt(2 * np.pi * t2)
    else:
        yv = np.zeros(400)
        for k in range(2):
            yv += 0.5 * np.exp(-0.5 * (xx - MU[k])**2 / SIG**2) / np.sqrt(2 * np.pi * SIG**2)
    ax[1].plot(xx, yv, ls=ls, lw=1.0, color="k", label=f"$s={s}$")
ax[1].set_xlabel("$x$"); ax[1].set_title("Marginals $p_s$", fontsize=9); ax[1].legend(fontsize=6, frameon=False)
plt.tight_layout(); plt.savefig("/tmp/work3/fig_traj.pdf"); plt.close()

# Fig 2: field v(x,s) and posterior mean at several s
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
xg = np.linspace(-2, 2, 800)
for s in [0.0, 0.3, 0.5, 0.7, 0.9]:
    ax[0].plot(xg, mean_post(xg, s), label=f"$s={s}$", lw=1.0)
ax[0].set_xlabel("$x$"); ax[0].set_ylabel(r"$m(x,s)$"); ax[0].set_title("Posterior mean of the endpoint", fontsize=9)
ax[0].legend(fontsize=6, frameon=False)
ax[1].plot(S, ent / np.log(2), color="k", label="posterior entropy (bits)")
ax[1].plot(S, lossfloor / lossfloor.max(), color="tab:red", label="loss floor (rescaled)")
ax[1].axvline(1 / (1 + A), color="k", ls=":", lw=0.8)
ax[1].set_xlabel("$s$"); ax[1].legend(fontsize=6, frameon=False); ax[1].set_title("Where the averaging is unavoidable", fontsize=9)
plt.tight_layout(); plt.savefig("/tmp/work3/fig_field.pdf"); plt.close()

# Fig 3: Euler outputs for several K
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=True)
for a_, K in zip(ax, [1, 4, 16, 256]):
    y = euler(K, x0[:60000])
    a_.hist(y, bins=120, range=(-2, 2), density=True, color="0.35")
    a_.set_title(f"$K={K}$", fontsize=9); a_.set_xlabel("$x_1$")
plt.tight_layout(); plt.savefig("/tmp/work3/fig_euler.pdf"); plt.close()


# ---- repeated seeds: uncertainty for last-step ratios and the W1 floor ----
def last_step_ratio(D, seed, n=200000):
    r = np.random.default_rng(seed); s = 1 - D
    k = r.integers(0, 2, n); x1 = MU[k] + SIG * r.standard_normal(n)
    xs = s * x1 + (1 - s) * r.standard_normal(n)
    y = mean_post(xs, s); pos = y[y > 0]; neg = y[y <= 0]
    return 0.5 * (pos.std() + neg.std()) / SIG
for D in [1/8, 1/16, 1/32, 1/64, 1/256]:
    vals = np.array([last_step_ratio(D, 100 + i) for i in range(20)])
    s = 1 - D; pred = s * SIG / np.sqrt(tau2(s))
    print("D", D, "mean", vals.mean().round(4), "sd", vals.std(ddof=1).round(4), "pred", pred.round(4))
w = []
for i in range(20):
    r = np.random.default_rng(500 + i)
    z0 = r.standard_normal(N)
    y = euler(2048, z0)
    k = r.integers(0, 2, N); t = MU[k] + SIG * r.standard_normal(N)
    w.append(w1(y, t))
print("W1 K=2048 over seeds mean", np.mean(w), "sd", np.std(w, ddof=1), "min", min(w), "max", max(w))
exact = SIG*np.sqrt(2/np.pi)*np.exp(-A**2/(2*SIG**2)) + A*(1-2*__import__("scipy.stats").stats.norm.cdf(-A/SIG))
print("E|x1| exact", exact)
print("s_c exact", 1/(1+np.sqrt(A**2-SIG**2)))
