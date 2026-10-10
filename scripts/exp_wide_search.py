"""Wider post-hoc search (144 settings) with nested leave-one-compound-out selection
over the whole space. Usage: python scripts/exp_wide_search.py results/wide_search.csv"""
import os
import sys, numpy as np, pandas as pd, itertools, time
ROOT = os.path.join(os.path.dirname(__file__), ".."); sys.path.insert(0, ROOT)
from shennong.profiles import build_profiles, l2, load_features, treatment_profiles
from shennong.moa import annotated, evaluate, NSCB_EXCLUDE, _masks
t0=time.time()
f = load_features(os.path.join(ROOT, "features"))
cache={}
def wells(nc):
    if nc not in cache:
        wmeta,_,B,_=build_profiles(f,nc); cache[nc]=(wmeta,B)
    return cache[nc]
wmeta,B=wells(128)
g=wmeta.groupby(["compound","conc"]).indices; keys=sorted(g)
tm_all,_=treatment_profiles(wmeta,B["fusion"]); nondmso=(tm_all.compound!="DMSO").values
ann_mask = tm_all.moa.notna().values & nondmso
def tagg(x,a):
    if a=="mean": return np.stack([x[g[k]].mean(0) for k in keys])
    if a=="median": return np.stack([np.median(x[g[k]],0) for k in keys])
    if a=="trim":  # mean of central wells (drop min/max norm when >=3)
        out=[]
        for k in keys:
            v=x[g[k]]
            if len(v)>=3:
                n=np.linalg.norm(v-np.median(v,0),axis=1); v=v[np.argsort(n)[:-1]]
            out.append(v.mean(0))
        return np.stack(out)
def sphere(P,frac):
    mu=P[nondmso].mean(0); X=P[nondmso]-mu; U,s,Vt=np.linalg.svd(X,full_matrices=False)
    k=int((np.cumsum(s**2)/np.sum(s**2)<frac).sum())+1
    return (P-mu)@Vt[:k].T/np.sqrt(s[:k]**2/len(X)+ (s[0]**2/len(X))*0.05)
CFG={}
for nc,w,a,post,blockwise in itertools.product([64,128,256],[0.4,0.7,1.0],["mean","trim"],["center","sphere95","sphere99","center_std"],[False,True]):
    wm,Bb=wells(nc)
    bl=[Bb["dino_cell"],Bb["dino_field"],Bb["classical"]]; wt=[1,1,w]
    def tr(P):
        mu=P[nondmso].mean(0)
        if post=="center": return P-mu
        if post=="center_std": return (P-mu)/(P[nondmso].std(0)+1e-6)
        if post=="sphere95": return sphere(P,0.95)
        if post=="sphere99": return sphere(P,0.99)
    if blockwise:   # aggregate + transform each view separately, then l2 and weight
        P=np.hstack([wt[i]*l2(tr(tagg(bl[i],a))) for i in range(3)])
    else:
        P=tr(tagg(np.hstack([wt[i]*l2(bl[i]) for i in range(3)]),a))
    CFG[(nc,w,a,post,blockwise)]=P
# reference configs
for nc,w in itertools.product([64,128,256],[0.4,0.7,1.0]):
    wm,Bb=wells(nc); W=np.hstack([l2(Bb["dino_cell"]),l2(Bb["dino_field"]),w*l2(Bb["classical"])])
    P=tagg(W,"mean"); CFG[(nc,w,"mean","center",False)]=P-P[nondmso].mean(0)
print("n configs",len(CFG), round(time.time()-t0,1),"s", flush=True)
rows=[]
for k,P in CFG.items():
    tm,p=annotated(tm_all,P); rows.append((*k,evaluate(tm,p)["accuracy"],evaluate(tm,p,batch_aware=True)["accuracy"]))
df=pd.DataFrame(rows,columns=["tvn","w","agg","post","blockwise","NSC","NSCB"])
print(df.groupby(["post","blockwise"])[["NSC","NSCB"]].agg(["median","max"]).round(3))
print(df.groupby(["agg"])[["NSC","NSCB"]].agg(["median","max"]).round(3))
print(df.sort_values("NSC",ascending=False).head(8).round(3).to_string(index=False))
print(df.sort_values("NSCB",ascending=False).head(5).round(3).to_string(index=False))
def nested(batch_aware):
    tm,_=annotated(tm_all,next(iter(CFG.values())))
    keep=~tm.moa.isin(NSCB_EXCLUDE).values if batch_aware else np.ones(len(tm),bool)
    tm=tm[keep].reset_index(drop=True); comp=tm.compound.values; moa=tm.moa.values; ban=_masks(tm,batch_aware)
    sims={}
    for k,P in CFG.items():
        p=l2(annotated(tm_all,P)[1][keep]); sims[k]=np.where(ban,-np.inf,p@p.T)
    names=list(sims); Sst=np.stack([sims[n] for n in names])  # C x n x n
    correct=[]
    for c in np.unique(comp):
        hold=comp==c; qi=np.where(~hold)[0]; qo=np.where(hold)[0]
        s=Sst[:,qi][:,:,~hold]; pred=moa[~hold][s.argmax(2)]
        best=(pred==moa[qi][None]).mean(1).argmax()
        correct+=list(moa[Sst[best][qo].argmax(1)]==moa[qo])
    return float(np.mean(correct))
print("NESTED over all", len(CFG), "configs: NSC", round(nested(False),4), "NSCB", round(nested(True),4), round(time.time()-t0,1),"s")
df.to_csv(sys.argv[1],index=False)
