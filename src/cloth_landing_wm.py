"""Action-conditioned terminal dynamics for one fixed carry/release skill."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest() if hasattr(hashlib,"file_digest") else hashlib.sha256(f.read()).hexdigest()


def features(rgb,q,cal,xy):
    # Action-relative rings in the table plane, observed before any carry command.
    eye,quat,K=cal
    rot=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
    angle=np.arange(16)*np.pi/8
    rings=np.array([.008,.032,.049,.061,.075])[:,None,None]*np.stack([np.cos(angle),np.sin(angle)],-1)
    pts=xy+rings.reshape(-1,2)
    cam=(np.c_[pts,np.full(len(pts),.0005)]-eye)@rot
    pix=cam@K.T
    u,v=np.rint(pix[:,:2]/pix[:,2:]).astype(int).T
    valid=(cam[:,2]>0)&(u>=0)&(v>=0)&(u<rgb.shape[1])&(v<rgb.shape[0])
    samples=np.zeros((len(v),3),dtype=np.int16)
    samples[valid]=rgb[v[valid],u[valid]]
    dark=(samples.max(1)>=55)&(samples.max(1)<=125)&(np.ptp(samples,axis=1)<=20)&valid
    red=(samples[:,0]-samples[:,1]>45)&(samples[:,0]-samples[:,2]>45)&valid
    local=np.stack((dark,red),axis=-1).reshape(5,16,2).mean(1).ravel()
    x=np.r_[local,valid.mean(),(xy-[-.13,-.23])/.10,np.sin(q)]
    assert x.shape==(19,) and np.isfinite(x).all()
    return x


def predict(model,x,xy,k=4,prior=False):
    dx=(model["X"]-x)**2
    d=dx[:,:10].mean(1)+.35*dx[:,10]+.008*dx[:,11:13].mean(1)+.008*dx[:,13:].mean(1)
    w=np.full(len(d),1./len(d))
    if not prior:
        ids=np.argsort(d,kind="stable")[:k]
        wk=np.exp(-d[ids]/max(.003,d[ids[-1]]*.6))
        if wk.sum()<=0:
            wk=np.ones(len(ids))
        w*=.01
        w[ids]+=.99*wk/wk.sum()
    # Y: landing-XY residual (mm), terminal z (mm), tilt, worst final disk margin (mm),
    # strict completion and three disjoint final center colors. Not the teacher's requested point.
    y=w@model["Y"]
    return dict(final_xy=(xy+y[:2]/1000).tolist(),z_mm=float(y[2]),tilt_deg=float(y[3]),
                margin_mm=float(y[4]),strict=float(y[5]),black=float(y[6]),red=float(y[7]),neither=float(y[8]))


def choose(model,rgb,q,cal,student,regions,settings):
    choices=[]
    for key,xy in [("student",student),*[(f"region{i}",v) for i,v in enumerate(regions)]]:
        if xy is None or any(np.linalg.norm(xy-np.array(c["xy"]))<.0005 for c in choices):
            continue
        x=features(rgb,q,cal,xy)
        pred=predict(model,x,xy,settings["neighbors"])
        score=pred["strict"]+.20*pred["black"]+.12*np.clip(pred["margin_mm"]/25.,-1.,1.)
        if key=="student":
            score+=settings["student_preference"]
        choices.append(dict(key=key,xy=np.asarray(xy).tolist(),forecast=pred,score=float(score)))
    viable=[c for c in choices if c["forecast"]["strict"]>=settings["min_strict"] and
            c["forecast"]["black"]>=settings["min_black"]]
    best=max(viable,key=lambda c:c["score"]) if viable else None
    return (np.array(best["xy"]) if best else None),dict(selected=best["key"] if best else "hold",choices=choices)


def load(path,expected):
    meta=json.loads((path/"freeze.json").read_text())
    assert sha(path/"landing.npz")==expected==meta["sha256"]
    assert sha(Path(__file__))==meta["model_code_sha256"]
    with np.load(path/"landing.npz",allow_pickle=False) as f:
        return dict(f)
