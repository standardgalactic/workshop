"""Illustrations for 'Counting Edges and Counting Equations'. Toy models, not a reproduction."""
import numpy as np, json
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(12); out={}
# ---- 1. conditioning of a dimension-matched property map: van der Pol with parameters (mu, omega)
def vdp_props(mu,om):
    f=lambda t,s:[s[1], mu*(1-s[0]**2)*s[1]-om**2*s[0]]
    sol=solve_ivp(f,[0,400/om],[2.0,0.0],rtol=1e-10,atol=1e-12,dense_output=True,max_step=0.02/om*min(1,5/mu) if mu>1 else 0.02/om)
    t=np.linspace(200/om,400/om,400000); x=sol.sol(t)[0]
    # period from successive upward zero crossings
    idx=np.where((x[:-1]<0)&(x[1:]>=0))[0]
    tz=t[idx]-x[idx]*(t[idx+1]-t[idx])/(x[idx+1]-x[idx])
    T=np.mean(np.diff(tz)[-5:]); A=x[t>=tz[-2]].max() if len(tz)>=2 else x.max()
    return A,T
def jac(mu,om,h=1e-3):
    P=lambda m,o: np.array(vdp_props(m,o))
    J=np.zeros((2,2))
    J[:,0]=(P(mu+h,om)-P(mu-h,om))/(2*h); J[:,1]=(P(mu,om+h)-P(mu,om-h))/(2*h)
    return J
rows=[]
for mu in [0.5,1.0,2.0]:
    J=jac(mu,1.0)
    # relative sensitivities (log-derivatives) so units do not matter
    A,T=vdp_props(mu,1.0); Jl=np.diag([1/A,1/T])@J@np.diag([mu,1.0])
    sv=np.linalg.svd(Jl,compute_uv=False)
    rows.append(dict(mu=mu,A=A,T=T,sv_ratio=float(sv[0]/sv[1]),dlogA_dlogmu=float(Jl[0,0]),dlogA_dlogom=float(Jl[0,1]),dlogT_dlogmu=float(Jl[1,0]),dlogT_dlogom=float(Jl[1,1])))
    print(rows[-1])
out["vdp"]=rows
# ---- 2. what the success statistic measures under the protocol of the network experiment
n=100; delta=0.05
def success_grid(a,g,deg=6,trials=60,step=5):
    mts=list(range(5,101,step)); mes=list(range(1,101,step))
    R=np.zeros((len(mts),len(mes)))
    for _ in range(trials):
        adj=(rng.random((n,n))<deg/n).astype(float); np.fill_diagonal(adj,0)
        perm=rng.permutation(n)
        for i,mt in enumerate(mts):
            T=perm[:mt]
            for j,me in enumerate(mes):
                pert=perm[:me]                      # perturbed nodes: targets first, then others (as in the protocol)
                resp=np.zeros(n); resp[pert]+=a      # direct effect of one incoming-edge change
                resp+= g*a*(adj[:,pert].sum(1))      # propagated effect from perturbed neighbours
                R[i,j]+=(resp[T]>delta).mean()
    return mts,mes,R/trials
res={}
for a,g,lab in [(0.08,0.0,"no propagation"),(0.08,0.3,"weak propagation"),(0.08,0.9,"strong propagation")]:
    mts,mes,R=success_grid(a,g)
    res[lab]=R
    k=mts.index(50)
    first=[mes[j] for j in range(len(mes)) if R[k,j]>=0.9]
    print(lab,"m_target=50: smallest m_edge with success rate>=0.9:",first[0] if first else None)
out["protocol"]={k:v.tolist() for k,v in res.items()}
json.dump(out,open("modul_results.json","w"),indent=1)
plt.rcParams.update({"font.size":8,"font.family":"serif"})
fig,ax=plt.subplots(1,3,figsize=(7.4,2.6))
for a_,(lab,R) in zip(ax,res.items()):
    a_.imshow(R,origin="lower",aspect="auto",extent=[mes[0],mes[-1],mts[0],mts[-1]],vmin=0,vmax=1,cmap="viridis")
    a_.plot([0,100],[0,100],"w--",lw=0.8); a_.set_title(lab,fontsize=8); a_.set_xlabel("$m_{edge}$")
ax[0].set_ylabel("$m_{target}$"); plt.tight_layout(); plt.savefig("fig_modul.pdf")
