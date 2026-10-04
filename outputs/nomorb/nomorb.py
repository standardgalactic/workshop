"""Illustrations for 'Where Covariance Comes From'. Toy models, not a reanalysis of ABCD data."""
import numpy as np, json, warnings
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
rng=np.random.default_rng(8); out={}
N=11868
KS,KX,MX,MY=0.7,0.3,4.5,5.5     # shared and specific gamma components (rate 1), total mean 1
def genA(n):
    gs=rng.gamma(KS,1.0,n)
    return rng.poisson(MX*(gs+rng.gamma(KX,1.0,n))), rng.poisson(MY*(gs+rng.gamma(KX,1.0,n)))
P0=0.75
def genB(n):
    sym=rng.random(n)>P0
    x=np.where(sym, rng.negative_binomial(1.0,1.0/(1+MX/(1-P0)),n),0)
    y=np.where(sym, rng.negative_binomial(1.0,1.0/(1+MY/(1-P0)),n),0)
    return x,y
def seg_fit(x,y,tau,link):
    res=[]
    for m in (x<tau, x>=tau):
        X=sm.add_constant(x[m].astype(float))
        fam=sm.families.Poisson() if link=="log" else sm.families.Gaussian()
        f=sm.GLM(y[m].astype(float),X,family=fam).fit()
        res.append((f.params[1],f.llf))
    return res
def changepoint(x,y,link):
    best=None
    for tau in range(2,13):
        if (x<tau).sum()<200 or (x>=tau).sum()<200: continue
        r=seg_fit(x,y,tau,link); ll=r[0][1]+r[1][1]
        if best is None or ll>best[0]: best=(ll,tau,r[0][0],r[1][0])
    return best[1:]
def covshare(x,y,tau):
    d=(x-x.mean())*(y-y.mean())
    return float(d[x<tau].sum()/d.sum()), float((x<tau).mean())
rows={}
for name,gen in [("A",genA),("B",genB)]:
    R=[]
    for _ in range(30):
        x,y=gen(N)
        tl,b1l,b2l=changepoint(x,y,"log")
        tg,b1g,b2g=changepoint(x,y,"id")
        cs,fr=covshare(x,y,tl)
        R.append([tl,b1l,b2l,np.exp(b1l)-1,np.exp(b2l)-1,b1g,b2g,cs,fr,np.corrcoef(x,y)[0,1]])
    R=np.array(R); m=R.mean(0)
    rows[name]=dict(tau=m[0],B1log=m[1],B2log=m[2],pct1=m[3],pct2=m[4],B1id=m[5],B2id=m[6],cov_share_below=m[7],frac_below=m[8],r=m[9])
    print(name,{k:round(float(v),3) for k,v in rows[name].items()})
out["fits"]=rows
curves={}
for name,gen in [("A",genA),("B",genB)]:
    x,y=gen(3_000_000)
    xs=np.arange(0,31)
    ey=np.array([y[x==k].mean() if (x==k).sum()>200 else np.nan for k in xs])
    curves[name]=ey
    print(name,"E[Y|X] at 0,1,2,5,10,20:",[round(float(ey[k]),2) for k in (0,1,2,5,10,20)])
out["curves"]={k:[None if np.isnan(v) else float(v) for v in vv] for k,vv in curves.items()}
json.dump(out,open("nomorb_results.json","w"),indent=1)
plt.rcParams.update({"font.size":9,"font.family":"serif"})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
xs=np.arange(0,31)
for name,ls,mk in [("A","-","o"),("B","--","s")]:
    ax[0].plot(xs,curves[name],ls,marker=mk,ms=3,color="k" if name=="A" else "tab:red",label={"A":"A: one shared liability","B":"B: asymptomatic subgroup"}[name])
ax[0].set_xlabel("internalizing count $X$"); ax[0].set_ylabel("$E[Y\\mid X]$"); ax[0].legend(fontsize=6,frameon=False); ax[0].set_xlim(0,25)
ax[0].set_title("Mean externalizing count, raw scale",fontsize=8)
for name,ls,mk in [("A","-","o"),("B","--","s")]:
    e=curves[name]; ls_=np.log(e[1:])-np.log(e[:-1])
    ax[1].plot(xs[:-1],ls_,ls,marker=mk,ms=3,color="k" if name=="A" else "tab:red")
ax[1].set_xlim(0,15); ax[1].set_ylim(-0.1,1.0); ax[1].set_xlabel("internalizing count $X$"); ax[1].set_ylabel("local log-scale slope")
ax[1].set_title("Both models: steep early, flat late",fontsize=8)
plt.tight_layout(); plt.savefig("fig_nomorb.pdf")
