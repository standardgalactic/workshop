"""Illustrations for 'A Proof of the Wrong Thing'.  Toy computations, not replications."""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("/tmp/work6", exist_ok=True)
rng = np.random.default_rng(5)
out = {}

# ---------- 1. CRB for the near-field ULA of the case study: known vs unknown path gain ----------
LAM = 0.1; D = LAM / 2

def steer(M, theta, R, ref="end"):
    m = np.arange(M, dtype=float)
    if ref == "center":
        m = m - (M - 1) / 2
    x = m * D
    ph = 2 * np.pi / LAM * (x * np.sin(theta) - x**2 * np.cos(theta)**2 / (2 * R))
    return np.exp(1j * ph)

def derivs(M, theta, R, ref, h=1e-6):
    a = lambda t, r: steer(M, t, r, ref)
    da_t = (a(theta + h, R) - a(theta - h, R)) / (2 * h)
    da_r = (a(theta, R + 1e-4) - a(theta, R - 1e-4)) / 2e-4
    return a(theta, R), da_t, da_r

def crb(M, theta, R, ref, known_gain, sigma2=1.0):
    a, dt, dr = derivs(M, theta, R, ref)
    cols = [dt, dr]
    if not known_gain:
        cols += [a, 1j * a]
    J = np.stack(cols, axis=1)           # |alpha| = 1, alpha = 1 at the reference
    F = 2 / sigma2 * np.real(J.conj().T @ J)
    C = np.linalg.inv(F)
    return C[0, 0], C[1, 1]

theta0 = np.deg2rad(30.0)
rows = []
print("CRB ratios (known gain, ref=end) / (unknown gain) and (known gain, ref=center)/(unknown gain)")
for M in [16, 32, 64]:
    for R in [10.0, 30.0]:
        kt_e, kr_e = crb(M, theta0, R, "end", True)
        kt_c, kr_c = crb(M, theta0, R, "center", True)
        ut_e, ur_e = crb(M, theta0, R, "end", False)
        ut_c, ur_c = crb(M, theta0, R, "center", False)
        rows.append(dict(M=M, R=R,
                         known_end_over_unknown_theta=kt_e / ut_e, known_end_over_unknown_R=kr_e / ur_e,
                         known_center_over_unknown_theta=kt_c / ut_c, known_center_over_unknown_R=kr_c / ur_c,
                         unknown_end_vs_center_theta=ut_e / ut_c, unknown_end_vs_center_R=ur_e / ur_c,
                         known_end_vs_center_theta=kt_e / kt_c))
        r = rows[-1]
        print("M=%d R=%g | theta: known/end %.3f  known/center %.3f | R: known/end %.3f known/center %.3f | unknown end/center theta %.4f"
              % (M, R, r["known_end_over_unknown_theta"], r["known_center_over_unknown_theta"],
                 r["known_end_over_unknown_R"], r["known_center_over_unknown_R"], r["unknown_end_vs_center_theta"]))
out["crb"] = rows

# ---------- 2. threshold effect: far-field single-snapshot angle estimation ----------
M = 16; th_true = np.deg2rad(20.0)
grid = np.deg2rad(np.linspace(-90, 90, 6001))
m = np.arange(M)
Agrid = np.exp(1j * np.pi * np.outer(m, np.sin(grid)))          # M x G
a0 = np.exp(1j * np.pi * m * np.sin(th_true))
da0 = 1j * np.pi * m * np.cos(th_true) * a0
# CRB(theta) with unknown complex gain, unit noise scaling: F = 2/sigma2 Re[J^H J]
J = np.stack([da0, a0, 1j * a0], axis=1)
Fu = 2 * np.real(J.conj().T @ J)
crb_theta_unit = np.linalg.inv(Fu)[0, 0]                         # at sigma^2 = 1, |alpha| = 1
snr_db = np.arange(-10, 26, 5)
res2 = []
for s in snr_db:
    sigma2 = 10 ** (-s / 10)
    n = 3000
    noise = (rng.standard_normal((n, M)) + 1j * rng.standard_normal((n, M))) * np.sqrt(sigma2 / 2)
    y = a0[None, :] + noise
    score = np.abs(y @ Agrid.conj())                              # n x G  (|a^H y|)
    est = grid[np.argmax(score, axis=1)]
    mse = np.mean((est - th_true) ** 2)
    crb_s = crb_theta_unit * sigma2
    res2.append((int(s), float(mse), float(crb_s), float(mse / crb_s)))
    print("SNR %3d dB  MSE %.3e  CRB %.3e  ratio %.2f" % (s, mse, crb_s, mse / crb_s))
out["threshold"] = res2

# ---------- 3. a verified theorem about a model: correlated fading ----------
def exp_corr(n, r):
    idx = np.arange(n)
    return r ** np.abs(idx[:, None] - idx[None, :])

def ergodic_capacity(nt, nr, rho_db, rc, N=40000):
    rho = 10 ** (rho_db / 10)
    Rt = exp_corr(nt, rc); Rr = exp_corr(nr, rc)
    Lt = np.linalg.cholesky(Rt); Lr = np.linalg.cholesky(Rr)
    G = (rng.standard_normal((N, nr, nt)) + 1j * rng.standard_normal((N, nr, nt))) / np.sqrt(2)
    H = Lr[None] @ G @ Lt.conj().T[None]
    HH = H @ np.conj(np.transpose(H, (0, 2, 1)))
    det = np.linalg.det(np.eye(nr)[None] + rho / nt * HH).real
    return float(np.mean(np.log2(det)))
cap = {}
for rc in [0.0, 0.5, 0.7, 0.9]:
    cap[rc] = ergodic_capacity(4, 4, 10, rc)
    print("4x4, 10 dB, correlation %.1f: ergodic capacity %.3f bit/s/Hz" % (rc, cap[rc]))
out["capacity"] = {str(k): v for k, v in cap.items()}

# ---------- 4. best-of-5 reporting ----------
print("best-of-5 success vs per-run success p")
bo = {}
for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
    bo[p] = 1 - (1 - p) ** 5
    print(p, round(bo[p], 3))
out["bestof5"] = {str(k): v for k, v in bo.items()}
json.dump(out, open("/tmp/work6/wireless_results.json", "w"), indent=1)

# ---------- figure ----------
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
s = [r[0] for r in res2]
ax[0].semilogy(s, [r[1] for r in res2], "-o", ms=3, color="k", label="ML estimator MSE")
ax[0].semilogy(s, [r[2] for r in res2], "--s", ms=3, color="tab:red", label="Cramér–Rao bound (unknown complex gain)")
ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("MSE of $\\hat\\theta$ (rad$^2$)")
ax[0].set_title("A correct bound, not the attainable error", fontsize=9); ax[0].legend(fontsize=6, frameon=False)
rcs = list(cap.keys())
ax[1].plot(rcs, [cap[r] for r in rcs], "-o", ms=3, color="k")
ax[1].set_xlabel("antenna correlation coefficient"); ax[1].set_ylabel("ergodic capacity (bit/s/Hz)")
ax[1].set_title("A verified theorem about one model", fontsize=9)
plt.tight_layout(); plt.savefig("/tmp/work6/fig_wireless.pdf"); plt.close()
