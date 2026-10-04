"""Toy factorial model for 'Same Backbone, Different Everything'. Illustration only."""
import numpy as np, json, itertools
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(9)
NP,NC,M_DEV,M_TEST,KP=300,70,128,178,10
def problem(T,sv):
    mu=0.75 if T else 0.70; f=0.05 if T else 0.30
    q=np.clip(rng.normal(mu,0.08,NC),0,1); fail=rng.random(NC)<f
    tau=rng.normal(0,0.05,M_DEV+M_TEST)
    eps=rng.standard_normal((NC,M_DEV+M_TEST))
    S=np.clip(q[:,None]+tau[None,:]+sv*eps,0,1); S[fail]=0.0
    return S[:,:M_DEV],S[:,M_DEV:]
def greedy_portfolio(dev,k):
    chosen=[]; cur=np.zeros(dev.shape[1])
    for _ in range(k):
        gains=np.maximum(dev,cur[None,:]).mean(1)
        gains[chosen]=-1; h=int(np.argmax(gains)); chosen.append(h); cur=np.maximum(cur,dev[h])
    return chosen
def cell(T,S,P,sv,reps=NP):
    out=[]
    for _ in range(reps):
        dev,test=problem(T,sv)
        if not S and not P: idx=[0]
        elif S and not P: idx=[int(np.argmax(dev.mean(1)))]
        elif not S and P: idx=list(range(KP))
        else: idx=greedy_portfolio(dev,KP)
        out.append(test[idx].max(0).mean())
    return float(np.mean(out)),float(np.std(out,ddof=1)/np.sqrt(reps))
res={}
for sv in [0.03,0.08,0.12]:
    cells={}
    for T,S,P in itertools.product([0,1],repeat=3):
        cells[(T,S,P)]=cell(T,S,P,sv)
    # Shapley over the three factors
    v=lambda s: cells[(int('T' in s),int('S' in s),int('P' in s))][0]
    fac=['T','S','P']; sh={}
    for f in fac:
        tot=0
        for perm in itertools.permutations(fac):
            i=perm.index(f); before=set(perm[:i]); tot+=v(before|{f})-v(before)
        sh[f]=tot/6
    orders={}
    for perm in [('T','S','P'),('P','S','T')]:
        cur=set(); inc={}
        for f in perm:
            inc[f]=v(cur|{f})-v(cur); cur|={f}
        orders["".join(perm)]=inc
    res[sv]=dict(cells={"".join(map(str,k)):vv for k,vv in cells.items()},shapley=sh,orders=orders,total=v({'T','S','P'})-v(set()))
    print("sigma_v",sv,"bare %.3f full %.3f"%(v(set()),v({'T','S','P'})))
    print(" cells (T,S,P):",{"".join(map(str,k)):round(vv[0],3) for k,vv in cells.items()})
    print(" shapley",{k:round(x,3) for k,x in sh.items()}," orders",{k:{a:round(b,3) for a,b in d.items()} for k,d in orders.items()})
json.dump({str(k):v for k,v in res.items()},open("attrib_results.json","w"),indent=1)
plt.rcParams.update({"font.size":9,"font.family":"serif"})
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
x=np.arange(3); w=0.25; sv=0.08
for j,(lab,key,col) in enumerate([("tools first","TSP","0.2"),("portfolio first","PST","tab:red")]):
    inc=res[sv]["orders"][key]
    ax[0].bar(x+(j-0.5)*w*1.2,[inc["T"],inc["S"],inc["P"]],w,color=col,label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels(["tools","dev selection","portfolio"]); ax[0].set_ylabel("score increment"); ax[0].legend(fontsize=6,frameon=False)
ax[0].set_title("Same gap, order-dependent increments",fontsize=8)
svs=[0.03,0.08,0.12]
ax[1].plot(svs,[res[s]["cells"]["011"][0]-res[s]["cells"]["010"][0] for s in svs],"-o",ms=3,color="k",label="portfolio over best single (tools off)")
ax[1].plot(svs,[res[s]["cells"]["111"][0]-res[s]["cells"]["110"][0] for s in svs],"--s",ms=3,color="tab:blue",label="portfolio over best single (tools on)")
ax[1].set_xlabel("instance-specific heuristic variation $\\sigma_v$"); ax[1].set_ylabel("portfolio effect"); ax[1].legend(fontsize=6,frameon=False)
ax[1].set_title("Portfolio effect scales with heterogeneity",fontsize=8)
plt.tight_layout(); plt.savefig("fig_attrib.pdf")
