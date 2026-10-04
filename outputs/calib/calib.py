"""Toy model of metric calibration under sparse perturbation signal.

G genes, k affected with shift +-d (in units of the single-cell SD, v=1).
Ground truth y = mean of n_gt cells, technical duplicate = mean of n_td other
cells, mean baseline = 0 (the shared reference).  Interpolated duplicate uses
alpha = 1 - BH-adjusted p from a z-test on the duplicate half, mirroring the
construction described by Miller et al. (2026); the weighted metric uses
weights 1 - BH-adjusted p from the ground-truth half.  Illustration only.
"""
import numpy as np, json, os
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("/home/bonobo/new/outputs/calib/", exist_ok=True)
rng = np.random.default_rng(11)
G, N_GT, N_TD, R = 8192, 100, 100, 300

def bh(p):
    m = p.size
    o = np.argsort(p)
    ranked = p[o] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(ranked, 1.0)
    return out

def one(k, d):
    mu = np.zeros(G)
    idx = rng.choice(G, k, replace=False)
    mu[idx] = d * rng.choice([-1, 1], k)
    y = mu + rng.standard_normal(G) / np.sqrt(N_GT)
    td = mu + rng.standard_normal(G) / np.sqrt(N_TD)
    z_td = td * np.sqrt(N_TD)
    a = 1 - bh(2 * norm.sf(np.abs(z_td)))
    idp = a * td
    z_gt = y * np.sqrt(N_GT)
    w = 1 - bh(2 * norm.sf(np.abs(z_gt)))
    w = w / w.mean() if w.sum() > 0 else np.ones(G)
    def mse(p): return np.mean((p - y) ** 2)
    def wmse(p): return np.mean(w * (p - y) ** 2)
    neg = np.zeros(G)
    r = dict(
        mse_n=mse(neg), mse_td=mse(td), mse_id=mse(idp), mse_or=mse(mu),
        w_n=wmse(neg), w_td=wmse(td), w_id=wmse(idp), w_or=wmse(mu))
    r["drf_mse_id"] = (r["mse_n"] - r["mse_id"]) / r["mse_n"]
    r["drf_mse_or"] = (r["mse_n"] - r["mse_or"]) / r["mse_n"]
    r["drf_w_id"] = (r["w_n"] - r["w_id"]) / r["w_n"]
    r["drf_w_or"] = (r["w_n"] - r["w_or"]) / r["w_n"]
    r["td_beats_mean"] = float(r["mse_td"] < r["mse_n"])
    return r

res = {}
for d in [1.0, 2.0]:
    for k in [3, 10, 30, 100, 300, 1000]:
        rows = [one(k, d) for _ in range(R)]
        res[f"d{d}_k{k}"] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
        res[f"d{d}_k{k}"]["drf_mse_id_sd"] = float(np.std([r["drf_mse_id"] for r in rows], ddof=1))
json.dump(res, open("/home/bonobo/new/outputs/calib/calib_results.json", "w"), indent=1)

for d in [1.0, 2.0]:
    print("delta =", d)
    for k in [3, 10, 30, 100, 300, 1000]:
        r = res[f"d{d}_k{k}"]
        print(k, "MSEn %.5f MSEtd %.5f MSEid %.5f | TDbeatsMean %.2f | DRF mse id %.3f ceil %.3f | DRF w id %.3f ceil %.3f" % (
            r["mse_n"], r["mse_td"], r["mse_id"], r["td_beats_mean"], r["drf_mse_id"], r["drf_mse_or"], r["drf_w_id"], r["drf_w_or"]))

# theory: TD beats mean iff k d^2 > G/N_TD (expected)
print("threshold k*d^2 > G/N_TD =", G / N_TD)

plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ks = [3, 10, 30, 100, 300, 1000]
for d, ls in [(1.0, "--"), (2.0, "-")]:
    ax[0].plot(ks, [res[f"d{d}_k{k}"]["mse_n"] for k in ks], ls, color="tab:blue", marker="o", ms=3, label=f"mean baseline, $\\delta$={d:g}")
    ax[0].plot(ks, [res[f"d{d}_k{k}"]["mse_td"] for k in ks], ls, color="tab:red", marker="s", ms=3, label=f"technical duplicate, $\\delta$={d:g}")
ax[0].set_xscale("log"); ax[0].set_xlabel("affected genes $k$"); ax[0].set_ylabel("MSE vs ground truth")
ax[0].legend(fontsize=6, frameon=False); ax[0].set_title("Positive control vs negative control", fontsize=9)
for d, ls in [(2.0, "-")]:
    ax[1].plot(ks, [res[f"d{d}_k{k}"]["drf_mse_id"] for k in ks], ls, color="k", marker="o", ms=3, label="MSE, interpolated duplicate")
    ax[1].plot(ks, [res[f"d{d}_k{k}"]["drf_mse_or"] for k in ks], "--", color="k", marker="o", ms=3, mfc="w", label="MSE, oracle ceiling")
    ax[1].plot(ks, [res[f"d{d}_k{k}"]["drf_w_id"] for k in ks], ls, color="tab:green", marker="s", ms=3, label="weighted, interpolated duplicate")
    ax[1].plot(ks, [res[f"d{d}_k{k}"]["drf_w_or"] for k in ks], "--", color="tab:green", marker="s", ms=3, mfc="w", label="weighted, oracle ceiling")
ax[1].set_xscale("log"); ax[1].set_xlabel("affected genes $k$"); ax[1].set_ylabel("DRF")
ax[1].set_ylim(0, 1); ax[1].legend(fontsize=6, frameon=False); ax[1].set_title("DRF and its attainable ceiling ($\\delta$=2)", fontsize=9)
plt.tight_layout(); plt.savefig("/home/bonobo/new/outputs/calib/fig_calib.pdf"); plt.close()

# ---- failure sensitivity: right on signal, wrong elsewhere ----
def garbage_case(k, d, sd=0.3):
    out = []
    for _ in range(R):
        mu = np.zeros(G)
        idx = rng.choice(G, k, replace=False)
        mu[idx] = d * rng.choice([-1, 1], k)
        y = mu + rng.standard_normal(G) / np.sqrt(N_GT)
        z_gt = y * np.sqrt(N_GT)
        w = 1 - bh(2 * norm.sf(np.abs(z_gt))); w = w / w.mean()
        g = mu + sd * rng.standard_normal(G)
        mse = lambda p: np.mean((p - y) ** 2)
        wm = lambda p: np.mean(w * (p - y) ** 2)
        out.append((mse(g) / mse(np.zeros(G)), wm(g) / wm(np.zeros(G)), mse(mu) / mse(np.zeros(G)), wm(mu) / wm(np.zeros(G))))
    return np.mean(out, axis=0)
gar = {}
print("failure sensitivity: score relative to mean baseline (1.0 = same as mean baseline, lower better)")
for k in [3, 10, 30, 100]:
    r = garbage_case(k, 2.0)
    gar[k] = [float(x) for x in r]
    print(k, "garbage: MSE ratio %.2f  weighted ratio %.2f | oracle: MSE ratio %.2f weighted ratio %.2f" % tuple(r))
res["garbage"] = gar
json.dump(res, open("/home/bonobo/new/outputs/calib/calib_results.json", "w"), indent=1)
