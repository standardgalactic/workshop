import numpy as np, json
rng=np.random.default_rng(20261010)
n=40; K=40; g=1.0/K; E=4; N=4000

def run(c,o,delta=0.0,help_=0.0,c_shuf_train=None,o_shuf_train=0.0):
    """Train on fixed order, test fix / shuffled / isolated. returns dict."""
    pc=1-(1-c)**E; po=1-(1-o)**E
    C=rng.random((N,n))<pc           # content trace
    O=rng.random((N,n))<po           # order trace keyed to trained predecessor
    # fixed order test: context matches
    def respond(content_ok, order_ok):
        ok=content_ok|order_ok
        guess=rng.random(ok.shape)<g
        return ok|guess
    Yfix=respond(C,O)
    # shuffled test: random permutation per learner; match prob for item j = pred equal
    Ysh=np.zeros((N,n),bool); match_sh=np.zeros((N,n),bool)
    for k in range(N):
        perm=rng.permutation(n)               # test order of items
        pred=np.full(n,-2); pred[perm[1:]]=perm[:-1]; pred[perm[0]]=-1   # -1 = start
        trained=np.arange(n)-1                # trained predecessor of item j is j-1 (item 0: start=-1)
        match_sh[k]=pred==trained
    disrupt=(rng.random((N,n))<delta)&(~match_sh)      # shuffle disrupts content retrieval
    helpc=(rng.random((N,n))<help_)&(~match_sh)        # accidental helpful cue for unknown items
    Ysh=respond(C&~disrupt, (O&match_sh)|helpc)
    Yiso=respond(C,np.zeros_like(C))
    # randomized match/mismatch test (Z), independent of item
    Z=rng.random((N,n))<0.5
    disZ=(rng.random((N,n))<delta)&(~Z)
    helpZ=(rng.random((N,n))<help_)&(~Z)
    Yz=respond(C&~disZ,(O&Z)|helpZ)
    out={}
    out['acc_fix']=Yfix.mean(); out['acc_shuf']=Ysh.mean(); out['acc_iso']=Yiso.mean()
    out['D_shuf']=Yfix.mean()-Ysh.mean(); out['D_iso']=Yfix.mean()-Yiso.mean()
    out['theta']=((~C)&O).mean()*(1-g)
    # test-time randomized contrast and plug-in CMI (bits), conditioning on item
    eff=Yz[Z].mean()-Yz[~Z].mean()
    cmi=0.0
    for j in range(n):
        y=Yz[:,j].astype(int); z=Z[:,j].astype(int)
        mi=0.0
        for yy in (0,1):
            for zz in (0,1):
                pj=np.mean((y==yy)&(z==zz)); py=np.mean(y==yy); pz=np.mean(z==zz)
                if pj>0: mi+=pj*np.log2(pj/(py*pz))
        cmi+=mi
    out['effect']=eff; out['cmi']=cmi/n
    return out

def train_gap(c,o,c_sh,Ep=E):
    """accuracy under matched train/test schedule: fixed vs shuffled training."""
    pcf=1-(1-c)**Ep; pof=1-(1-o)**Ep; pcs=1-(1-c_sh)**Ep
    Cf=rng.random((N,n))<pcf; Of=rng.random((N,n))<pof
    Cs=rng.random((N,n))<pcs
    gu=lambda m: m|(rng.random(m.shape)<g)
    return gu(Cf|Of).mean(), gu(Cs).mean()

res={}
res['A']=run(0.30,0.0)
res['B']=run(0.10,0.35)
res['C']=run(0.20,0.20)
res['D']=run(0.30,0.0,delta=0.25)
res['E']=run(0.30,0.0,help_=0.10)
# CMI under a fixed training schedule: context is a function of item => zero
# training-gap non-identification
gA=train_gap(0.20,0.15,0.20)      # setting a: shuffling does not change learning, order cue present
gB=train_gap(0.20,0.0,0.20*1.0)  # placeholder
# find c_sh so that gap in setting b (o=0) matches setting a
a_fix,a_shuf=gA
target=a_fix-a_shuf
# setting b: o=0, same c for fixed, lower c_sh
best=None
for c_sh in np.linspace(0.05,0.2,61):
    f,s=train_gap(0.20,0.0,c_sh)
    if best is None or abs((f-s)-target)<abs(best[3]-target): best=(c_sh,f,s,f-s)
res['gap_a']={'fix':a_fix,'shuf':a_shuf,'gap':target}
res['gap_b']={'c_sh':best[0],'fix':best[1],'shuf':best[2],'gap':best[3]}
json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in res.items()},open('res.json','w'),indent=1)
for k,v in res.items(): print(k,{kk:round(float(vv),4) for kk,vv in v.items()})
