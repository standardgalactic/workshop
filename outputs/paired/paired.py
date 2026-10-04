"""Overlap rule vs paired inference, and the unit of replication.  Illustration only."""
import numpy as np, json, os
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("/home/bonobo/new/outputs/paired/", exist_ok=True)
rng = np.random.default_rng(3)

# ---------- analytic: effective size of the overlap rule ----------
rows = []
for rho in [0.0, 0.5, 0.8, 0.95]:
    thr = 3.92 / np.sqrt(2 * (1 - rho))        # z threshold on the difference
    size = 2 * stats.norm.sf(thr)
    rows.append((rho, thr, size))
    print("rho %.2f  effective z threshold %.2f  size of overlap rule %.2e" % (rho, thr, size))
print("unpaired normal threshold 1.96*sqrt2 =", 1.96 * np.sqrt(2))

# ---------- simulation 1: power of three rules, shared scene difficulty ----------
N, SIG, R = 200, 1.0, 4000

def trial(d, tau):
    s = tau * rng.standard_normal(N)
    a = d + s + SIG * rng.standard_normal(N)
    b = s + SIG * rng.standard_normal(N)
    # marginal 95% t intervals of each mean
    tc = stats.t.ppf(0.975, N - 1)
    ha = tc * a.std(ddof=1) / np.sqrt(N); hb = tc * b.std(ddof=1) / np.sqrt(N)
    overlap = (a.mean() - ha) > (b.mean() + hb) or (b.mean() - hb) > (a.mean() + ha)
    unp = stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05
    pai = stats.ttest_rel(a, b).pvalue < 0.05
    return (not (not overlap)), unp, pai, ha

def power(d, tau):
    r = np.array([trial(d, tau) for _ in range(R)], dtype=float)
    return r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean()

tab = {}
print("\nSimulation 1 (N=200, sigma=1):  columns: overlap-rule, unpaired t, paired t, CI half-width")
for tau in [0.0, 1.0, 2.0, 4.0]:
    rho = tau**2 / (tau**2 + SIG**2)
    for d in [0.0, 0.4]:
        p = power(d, tau)
        tab[f"tau{tau}_d{d}"] = dict(rho=rho, overlap=p[0], unpaired=p[1], paired=p[2], half=p[3])
        print("tau %.0f rho %.2f d %.1f : %.3f %.3f %.3f  half-width %.3f" % (tau, rho, d, *p))

# figure: power curves at tau=2
ds = np.linspace(0, 0.8, 9)
curves = {"overlap": [], "unpaired": [], "paired": []}
for d in ds:
    p = power(d, 2.0)
    curves["overlap"].append(p[0]); curves["unpaired"].append(p[1]); curves["paired"].append(p[2])

# ---------- simulation 2: unit of replication ----------
OMEGA, TAU = 0.1, 2.0
def fixed_checkpoint_rejection(n_scenes, reps=3000):
    rej = 0
    for _ in range(reps):
        eta_a, eta_b = OMEGA * rng.standard_normal(2)       # one training run per method, true method effect 0
        s = TAU * rng.standard_normal(n_scenes)
        a = eta_a + s + SIG * rng.standard_normal(n_scenes)
        b = eta_b + s + SIG * rng.standard_normal(n_scenes)
        rej += stats.ttest_rel(a, b).pvalue < 0.05
    return rej / reps
t3 = {}
print("\nSimulation 2: paired scene test, one training run per method, true method effect 0, omega=0.1")
for n in [50, 200, 1000, 5000, 20000]:
    t3[n] = fixed_checkpoint_rejection(n, 2000 if n < 5000 else 600)
    print(n, t3[n])

def seed_test(d, K=5, n_scenes=200, reps=3000):
    rej = 0
    for _ in range(reps):
        s = TAU * rng.standard_normal(n_scenes)           # same scenes for all runs
        A = np.array([d + OMEGA * rng.standard_normal() + (s + SIG * rng.standard_normal(n_scenes)).mean() for _ in range(K)])
        B = np.array([OMEGA * rng.standard_normal() + (s + SIG * rng.standard_normal(n_scenes)).mean() for _ in range(K)])
        rej += stats.ttest_ind(A, B, equal_var=False).pvalue < 0.05
    return rej / reps
sd = {d: seed_test(d) for d in [0.0, 0.1, 0.2, 0.3]}
print("seed-replicated test (K=5 runs per method), rejection by true effect:", sd)

# fixed-checkpoint test power when true effect is 0.2 (for comparison, it also rejects when effect is 0)
json.dump(dict(analytic=rows, sim1=tab, curves=dict(ds=ds.tolist(), **{k: [float(x) for x in v] for k, v in curves.items()}),
               fixed={str(k): v for k, v in t3.items()}, seeds={str(k): v for k, v in sd.items()}),
          open("/home/bonobo/new/outputs/paired/paired_results.json", "w"), indent=1, default=float)

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].plot(ds, curves["paired"], "-o", ms=3, color="k", label="paired $t$ test")
ax[0].plot(ds, curves["unpaired"], "--s", ms=3, color="tab:blue", label="unpaired $t$ test")
ax[0].plot(ds, curves["overlap"], ":^", ms=3, color="tab:red", label="95% intervals do not overlap")
ax[0].set_xlabel("true difference $d$"); ax[0].set_ylabel("rejection rate"); ax[0].set_title("Shared scene difficulty ($\\tau=2$)", fontsize=9)
ax[0].legend(fontsize=6, frameon=False)
ns = list(t3.keys())
ax[1].plot(ns, [t3[n] for n in ns], "-o", ms=3, color="k")
ax[1].axhline(0.05, color="0.5", lw=0.8, ls=":")
ax[1].set_xscale("log"); ax[1].set_xlabel("test scenes $N$"); ax[1].set_ylabel("rejection rate")
ax[1].set_title("True method effect zero, one run each", fontsize=9)
plt.tight_layout(); plt.savefig("/home/bonobo/new/outputs/paired/fig_paired.pdf"); plt.close()
