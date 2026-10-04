"""Pooled-cohort correlation illustrations for 'What a Correlation of 0.18 Carries'. Toy models."""
import numpy as np, json
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(11); out={}
# --- reconstruction of n and Fisher comparison from reported Spearman R and P
def n_from(r,p):
    t=stats.t.isf(p/2,1000)  # placeholder, refine by solving
    from scipy.optimize import brentq
    f=lambda n: 2*stats.t.sf(abs(r)*np.sqrt((n-2)/(1-r**2)),n-2)-p
    return brentq(f,5,5000)
n1=n_from(0.18,0.00045); n2=n_from(-0.061,0.18)
z=(np.arctanh(0.18)-np.arctanh(-0.061))/np.sqrt(1/(n1-3)+1/(n2-3))
print("implied n: %.0f and %.0f, total %.0f; Fisher z=%.2f p=%.2g; R^2=%.3f"%(n1,n2,n1+n2,z,2*stats.norm.sf(z),0.18**2))
out["recon"]=dict(n1=n1,n2=n2,z=float(z),p=float(2*stats.norm.sf(z)))
# --- pooled correlation from between-cohort covariation only
def sim(K,share,rho_b,N=374,reps=3000):
    Rs=[];rej=0
    for _ in range(reps):
        sizes=rng.multinomial(N-K*10,np.ones(K)/K)+10
        cid=np.repeat(np.arange(K),sizes)
        m=rng.multivariate_normal([0,0],[[1,rho_b],[rho_b,1]],K)*np.sqrt(share)
        x=m[cid,0]+rng.standard_normal(N)*np.sqrt(1-share)
        y=m[cid,1]+rng.standard_normal(N)*np.sqrt(1-share)
        r,p=stats.spearmanr(x,y); Rs.append(r); rej+=(p<0.05)
    return float(np.mean(Rs)),float(np.std(Rs)),rej/reps
tab={}
for K in [4,9]:
    for share in [0.1,0.3]:
        for rho_b in [0.0,0.6]:
            mR,sR,rej=sim(K,share,rho_b)
            tab[f"K{K}_s{share}_rho{rho_b}"]=(mR,sR,rej)
            print("K=%d share=%.1f rho_b=%.1f: mean pooled R %.3f (sd %.3f), naive rejection at 0.05: %.3f"%(K,share,rho_b,mR,sR,rej))
out["pooled"]=tab
json.dump(out,open("pooled_results.json","w"),indent=1)
plt.rcParams.update({"font.size":9,"font.family":"serif"})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
# illustrative scatter: 4 cohorts, no within-cohort association
K=4;N=374;share=0.3;sizes=np.array([140,100,80,54]);cid=np.repeat(np.arange(K),sizes)
m=np.array([[-0.9,-0.8],[-0.2,0.1],[0.3,0.5],[0.8,0.9]])*1.0
x=m[cid,0]*0.7+rng.standard_normal(N)*0.9; y=m[cid,1]*0.7+rng.standard_normal(N)*0.9
cols=["tab:blue","tab:orange","tab:green","tab:red"]
for k in range(K): ax[0].scatter(x[cid==k],y[cid==k],s=3,color=cols[k],alpha=0.6)
ax[0].set_xlabel("$S.\\ copri$ (rank score)"); ax[0].set_ylabel("Enterobacteriaceae richness (rank score)")
r_all=stats.spearmanr(x,y)[0]; rw=[stats.spearmanr(x[cid==k],y[cid==k])[0] for k in range(K)]
ax[0].set_title("Pooled R=%.2f; within-cohort R about 0"%r_all,fontsize=8); print("scatter pooled",r_all,"within",rw)
Ks=[4,9]
ax[1].bar([0,1],[tab["K4_s0.3_rho0.0"][2],tab["K9_s0.3_rho0.0"][2]],color="0.3")
ax[1].axhline(0.05,color="tab:red",ls="--"); ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["4 cohorts","9 cohorts"])
ax[1].set_ylabel("naive test rejection rate"); ax[1].set_title("True null; cohort share of variance 0.3",fontsize=8)
plt.tight_layout(); plt.savefig("fig_pooled.pdf")
