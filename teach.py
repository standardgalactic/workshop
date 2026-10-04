"""Illustrations for 'Privileged Teachers'. Toy model, not a replication."""
import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(21); out={}
# ---- 1. two-option example: oracle-label frequency vs expected value
G,g=10.0,3.0
rows=[]
for q in [0.2,0.3,0.4,0.5,0.7]:
    ev_east=q*G; ev_west=g
    rows.append(dict(q=q,P_east_best=q,EV_east=ev_east,EV_west=ev_west,
        label_rule="east" if q>0.5 else "west", ev_rule="east" if ev_east>ev_west else "west"))
    print(rows[-1])
out["two_option"]=rows
# ---- 2. mode selection: three scoring rules, K modes, scene-level unobserved path noise
def trial(K=4, rho=0.4, M=4000):
    mu=rng.normal(0,1,K); sig=rng.uniform(0.2,2.0,K); n=rng.integers(2,17,K)
    # scene draws: path gains g_{m,j}=mu+sig*eps; reranker picks argmax of rho*eps+sqrt(1-rho^2)*eta
    best=np.zeros(K); emax=np.zeros(K); execv=np.zeros(K)
    win=np.zeros(K)
    for m in range(K):
        pass
    gm=[];em=[];xm=[]
    for m in range(K):
        eps=rng.standard_normal((M,n[m])); eta=rng.standard_normal((M,n[m]))
        s=rho*eps+np.sqrt(1-rho**2)*eta
        j=np.argmax(s,axis=1)
        gains=mu[m]+sig[m]*eps
        gm.append(gains.max(1)); em.append(gains.max(1).mean()); xm.append(gains[np.arange(M),j].mean())
    gm=np.array(gm)             # K x M, best-in-mode gain per scene
    P=np.bincount(np.argmax(gm,axis=0),minlength=K)/M   # P(mode contains overall best path)
    em=np.array(em); xm=np.array(xm)
    v=xm.max()
    return dict(reg_label=v-xm[np.argmax(P)], reg_emax=v-xm[np.argmax(em)],
                dis_label=float(np.argmax(P)!=np.argmax(xm)), dis_emax=float(np.argmax(em)!=np.argmax(xm)))
res={}
for rho in [0.0,0.4,0.7,0.9]:
    R=[trial(rho=rho) for _ in range(400)]
    res[rho]={k:float(np.mean([r[k] for r in R])) for k in R[0]}
    res[rho]["se_reg_emax"]=float(np.std([r["reg_emax"] for r in R],ddof=1)/np.sqrt(len(R)))
    res[rho]["se_reg_label"]=float(np.std([r["reg_label"] for r in R],ddof=1)/np.sqrt(len(R)))
    print(rho,res[rho])
out["modes"]={str(k):v for k,v in res.items()}
# E[max of n std normals]
from scipy.stats import norm
cn={n:float(np.mean(rng.standard_normal((200000,n)).max(1))) for n in [2,4,8,16]}
print(cn); out["cn"]=cn
json.dump(out,open("teach_results.json","w"),indent=1)
plt.rcParams.update({"font.size":9,"font.family":"serif"})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
qs=np.linspace(0,1,101)
ax[0].plot(qs,qs*G,"k",label="expected gain of east"); ax[0].axhline(g,color="tab:red",ls="--",label="west (certain)")
ax[0].axvline(0.5,color="0.6",ls=":"); ax[0].axvline(g/G,color="0.6",ls=":")
ax[0].set_xlabel("probability east is open $q$"); ax[0].set_ylabel("gain"); ax[0].legend(fontsize=6,frameon=False)
ax[0].set_title("Label rule flips at $q=0.5$, value rule at $q=0.3$",fontsize=8)
rh=[0.0,0.4,0.7,0.9]
ax[1].plot(rh,[res[r]["reg_emax"] for r in rh],"-o",ms=3,color="k",label="rank by expected best-in-mode gain")
ax[1].plot(rh,[res[r]["reg_label"] for r in rh],"--s",ms=3,color="tab:blue",label="rank by P(mode holds the best path)")
ax[1].set_xlabel("reranker resolution $\\rho$"); ax[1].set_ylabel("mean regret"); ax[1].legend(fontsize=6,frameon=False)
ax[1].set_title("Regret vs executed-path value",fontsize=8)
plt.tight_layout(); plt.savefig("fig_teach.pdf")
