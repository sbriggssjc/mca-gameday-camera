from __future__ import annotations
import json, pathlib, numpy as np, sys

CLASSES=["offense","defense","special_teams"]
FEATS=["black_ratio","white_ratio","mag_med","mag_p90","vx_med","vy_med","vy_std","color_lead","n1","a1","n4","a4"]

def load_feats(out):
    feat=json.loads((out/"features.json").read_text())
    return feat

def load_seeds(out):
    seeds=json.loads((out/"seed_labels.json").read_text())
    return seeds

def build_XY(feat, seeds):
    X=[]; y=[]
    for src,lab in seeds.items():
        f=feat.get(src); 
        if not f or not f.get("ok"): continue
        X.append([float(f.get(k,0.0)) for k in FEATS])
        y.append(CLASSES.index(lab))
    X=np.array(X, dtype=np.float32); y=np.array(y, dtype=np.int64)
    return X,y

def normalize(X):
    mu=X.mean(axis=0); sigma=X.std(axis=0)+1e-6
    return (X-mu)/sigma, mu, sigma

def softmax(z):
    z=z - z.max(axis=1, keepdims=True)
    e=np.exp(z); return e/np.sum(e,axis=1,keepdims=True)

def train_softmax(X,y,lr=0.05,epochs=400,lam=1e-3):
    N,D=X.shape; C=len(CLASSES)
    W=np.zeros((D,C), dtype=np.float32); b=np.zeros((C,), dtype=np.float32)
    Y=np.eye(C, dtype=np.float32)[y]
    for _ in range(epochs):
        S=X.dot(W)+b
        P=softmax(S)
        gradW = X.T.dot(P - Y)/N + lam*W
        gradb = (P - Y).mean(axis=0)
        W -= lr*gradW; b -= lr*gradb
    return W,b

def save_model(out, mu,sigma,W,b):
    model={"mu":mu.tolist(),"sigma":sigma.tolist(),"W":W.tolist(),"b":b.tolist(),"classes":CLASSES,"feats":FEATS}
    (out/"side_model.json").write_text(json.dumps(model, indent=2))
    print("[train] wrote", out/"side_model.json")

def main(out_dir):
    out=pathlib.Path(out_dir)
    feat=load_feats(out); seeds=load_seeds(out)
    X,y=build_XY(feat,seeds)
    if len(y)<4:
        print("[train] need at least 4 seed examples across classes"); return
    Xn,mu,sigma=normalize(X)
    W,b=train_softmax(Xn,y,lr=0.05,epochs=800,lam=1e-3)
    save_model(out,mu,sigma,W,b)

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "output")
