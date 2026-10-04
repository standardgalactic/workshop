import itertools, math
def shapley(players, v):
    n=len(players); phi={p:0.0 for p in players}
    for perm in itertools.permutations(players):
        cur=frozenset()
        for p in perm:
            nxt=cur|{p}
            phi[p]+= v[nxt]-v[cur]
            cur=nxt
    f=math.factorial(n)
    return {p:phi[p]/f for p in players}
def loo(players,v):
    full=frozenset(players)
    return {p: v[full]-v[full-{p}] for p in players}
def stand(players,v):
    return {p: v[frozenset([p])]-v[frozenset()] for p in players}
def seq(order,v):
    cur=frozenset(); out={}
    for p in order:
        nxt=cur|{p}; out[p]=v[nxt]-v[cur]; cur=nxt
    return out
E=frozenset
# toy 1: weak individually, strong together
v1={E():0,E('A'):1,E('B'):1,E('AB'):10}
# toy 2: AND gate
v2={E():0,E('A'):0,E('B'):0,E('AB'):10}
for name,v in [('weak-solo',v1),('AND',v2)]:
    print(name,'seqAB',seq('AB',v),'seqBA',seq('BA',v),'shap',shapley('AB',v),'loo',loo('AB',v),'stand',stand('AB',v))
# paper chain
M,C,H='M','C','H'
base=61.81
def mk(extra):
    v={E():base,E([M]):66.21,E([M,C]):67.60,E([M,C,H]):68.45}
    v.update(extra); return v
# completion A: additive
vA=mk({E([C]):base+1.39,E([H]):base+0.85,E([C,H]):base+1.39+0.85,E([M,H]):66.21+0.85})
# completion B: C,H inert without M
vB=mk({E([C]):base,E([H]):base,E([C,H]):base,E([M,H]):66.21+0.85})
# completion C: H alone works, C inert alone, big M-H synergy
vC=mk({E([C]):base,E([H]):base+1.5,E([C,H]):base+1.5,E([M,H]):67.30})
for name,v in [('A',vA),('B',vB),('C',vC)]:
    s=shapley([M,C,H],v); tot=sum(s.values())
    print(name,{k:round(x,3) for k,x in s.items()},'sum',round(tot,3),'share',{k:round(x/tot*100,1) for k,x in s.items()},'loo',{k:round(x,2) for k,x in loo([M,C,H],v).items()},'stand',{k:round(x,2) for k,x in stand([M,C,H],v).items()})
print('seq MCH',seq([M,C,H],vA))
# metric nonlinearity: cm error 10.22,9.64,9.36,9.11 ; completion-error relative
