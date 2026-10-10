#!/usr/bin/env python3
"""
simulate_open_items.py

Dimensionless illustration of attention lock-in on one intractable item
among concurrent tractable demands. Not calibrated to any data.

    d' = -gamma*d + s + eta*d*w ,   w = d/(d+M)

d      demand (salience) of the intractable item
M      total demand of concurrent tractable items
gamma  decay / resolution rate
eta    feedback gain of attention on salience
s      constant inflow of salience
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
s, gamma, eta = 0.10, 0.50, 0.80
M_crit = s * (np.sqrt(eta) + np.sqrt(eta - gamma)) ** 2 / gamma ** 2

def roots(M):
    a = eta - gamma; b = s - gamma * M; c = s * M
    D = b * b - 4 * a * c
    if D < 0: return None
    r = np.sqrt(D)
    return (-b - r) / (2 * a), (-b + r) / (2 * a)

def run(M, shock_t, shock, T=120.0, dt=0.02):
    n = int(T / dt)
    d = 0.0
    t_arr, w_arr = [], []
    for k in range(n):
        t = k * dt
        if abs(t - shock_t) < dt / 2: d += shock
        w = d / (d + M) if d + M > 0 else 0.0
        d += dt * (-gamma * d + s + eta * d * w)
        t_arr.append(t); w_arr.append(d / (d + M))
        if d > 1e6: break
    return np.array(t_arr), np.array(w_arr)

cases = [(0.4, "M = 0.4 (below threshold)"),
         (1.2, "M = 1.2 (above threshold, small basin)"),
         (3.0, "M = 3.0 (above threshold, large basin)")]
shock_t, shock = 40.0, 2.0

print(f"M_crit = {M_crit:.4f}")
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for M, label in cases:
    r = roots(M)
    print(M, "roots:", None if r is None else tuple(round(x, 4) for x in r))
    t, w = run(M, shock_t, shock)
    ax.plot(t, w, label=label)
ax.axvline(shock_t, linestyle="--", alpha=0.5, label="Salience shock (+2.0)")
ax.set_xlabel("Simulation time")
ax.set_ylabel("Attention share of the intractable item")
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="center right", fontsize=8)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "fig_lockin.png", dpi=220)
plt.close(fig)

# numerical checks of the analytic claims
# (a) eta < gamma: unique stable equilibrium, share ~ s/(gamma*M) for large M
g2, e2 = 0.5, 0.3
for M in (5.0, 20.0, 100.0):
    a = e2 - g2; b = s - g2 * M; c = s * M
    D = b * b - 4 * a * c
    r = [(-b - np.sqrt(D)) / (2 * a), (-b + np.sqrt(D)) / (2 * a)]
    dstar = [x for x in r if x > 0][0]
    print(f"eta<gamma M={M}: d*={dstar:.4f} share={dstar/(dstar+M):.5f} approx s/(gM)={s/(g2*M):.5f}")

# (b) effective sample size under exchangeable correlation
for rho in (0.0, 0.1, 0.5, 1.0):
    n = 100
    print(f"rho={rho}: n_eff={n/(1+(n-1)*rho):.3f}, limit 1/rho={'inf' if rho==0 else 1/rho}")
