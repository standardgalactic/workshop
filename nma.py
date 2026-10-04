"""Toy network for 'Which Trials Count'. Illustration only; not data from any review."""
import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
# parameters theta=(F-P, N-P); edges: (design row, y, se)
def fit(fn_y, fn_se=2.5, include=True, yFP=-2.8, sFP=0.65, yNP=-2.0, sNP=2.0):
    rows=[([1,0],yFP,sFP),([0,1],yNP,sNP)]
    if include: rows.append(([1,-1],fn_y,fn_se))
    X=np.array([r[0] for r in rows],float); y=np.array([r[1] for r in rows]); w=1/np.array([r[2] for r in rows])**2
    V=np.linalg.inv(X.T@(w[:,None]*X)); th=V@X.T@(w*y)
    return th,np.sqrt(np.diag(V))
out={}
base,_=fit(-0.8,include=False); print("without F-N trial: F-P = %.2f"%base[0])
rep=[-0.8,-3,-5,-8,-12]
rows=[]
for r in rep:
    th,se=fit(r)
    # node-splitting for F-N: direct r vs indirect (thF - thN from the two P-edges)
    ind=(-2.8)-(-2.0); sind=np.sqrt(0.65**2+2.0**2); z=(r-ind)/np.sqrt(sind**2+2.5**2)
    rows.append(dict(reported=r,FP=float(th[0]),se=float(se[0]),shift=float(th[0]+2.8),p_inconsistency=float(2*norm.sf(abs(z)))))
    print(rows[-1])
out["rows"]=rows
# SMD arithmetic: same raw mean difference, different SD of change; variance of SMD with n per arm
def smd_var(d,n): return 2/n+d**2/(4*n)
tab=[]
for sd in [12,6,3]:
    d=8/sd; v=smd_var(d,20); tab.append((sd,d,v**0.5,1/v))
    print("SD change",sd,"d=%.2f se=%.2f weight=%.1f"%(d,v**0.5,1/v))
d0=0.3; print("typical d=0.3, n=20/arm: se=%.2f weight=%.1f"%(smd_var(d0,20)**0.5,1/smd_var(d0,20)))
out["smd"]=tab
json.dump(out,open("nma_results.json","w"),indent=1)
plt.rcParams.update({"font.size":9,"font.family":"serif"})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
xs=np.linspace(-14,-0.8,60)
sweep={}
for sfp,ls in [(0.4,":"),(0.65,"-"),(1.0,"--"),(1.5,"-.")]:
    fp=[fit(x,yFP=-2.8,sFP=sfp)[0][0]+2.8 for x in xs]; ax[0].plot(xs,fp,"k",ls=ls,label="direct F$-$P SE %.2f"%sfp)
    sweep[sfp]=float(fit(-12,sFP=sfp)[0][0]+2.8); print("direct SE",sfp,"shift at reported -12: %.2f"%sweep[sfp])
out["sweep"]={str(k):v for k,v in sweep.items()}
ax[0].set_xlabel("reported F$-$N difference in the single trial"); ax[0].set_ylabel("shift in F$-$P estimate"); ax[0].legend(fontsize=6,frameon=False)
ax[0].set_title("Shift in F$-$P from one F$-$N trial",fontsize=8)
ds=np.linspace(0.1,5,60)
ax[1].plot(ds,1/smd_var(ds,20),"k"); ax[1].set_xlabel("standardized effect $d$ ($n=20$ per arm)"); ax[1].set_ylabel("inverse-variance weight")
ax[1].set_title("Larger reported effects are down-weighted only mildly",fontsize=8)
plt.tight_layout(); plt.savefig("fig_nma.pdf")
