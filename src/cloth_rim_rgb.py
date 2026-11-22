"""Upright-can localization from calibrated RGB and known radius/height in the pickup workspace."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

K = np.array([[320*22/28,0,160],[0,320*22/28,120],[0,0,1.]])
NOMINAL = np.array([.095,-.265])
ANGLE = np.arange(96)*2*np.pi/96
RING = .05*np.stack([np.cos(ANGLE),np.sin(ANGLE)],axis=-1)


def from_poses(eye, gripper):
    z = np.array([-.05,-.265,.04])-eye
    z /= np.linalg.norm(z)
    x = np.cross(z,[0.,0.,1.])
    x /= np.linalg.norm(x)
    g = Rotation.from_quat(gripper[[4,5,6,3]])
    w = g*Rotation.from_quat([-.912179,-.0451242,.0486914,-.404379])
    return ((eye,np.stack([x,np.cross(z,x),z],axis=1),K),
            (gripper[:3]+g.apply([-.001,.1,-.04]),w.as_matrix(),K))


def project(points, cam):
    org, rot, intr = cam
    p = (points-org)@rot
    uv = p@intr.T
    return uv[...,:2]/np.maximum(.001,uv[...,2:]), p[...,2]


def rings(xy, height, cam):
    pts = xy[...,None,:]+RING
    return project(np.concatenate([pts,np.full((*pts.shape[:-1],1),height)],axis=-1),cam)


def estimate(overhead, wrist, cams):
    dark = overhead.max(2)<55
    r,g,b = overhead.astype(int).transpose(2,0,1)
    yellow = (r>b+35)&(g>b+35)
    comps,_,stats,cent = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    likely = []
    for i in range(1,comps):
        if 300<=stats[i,4]<=4800:
            uv = np.r_[cent[i],1.]
            org,rot,intr = cams[0]
            ray = rot@np.linalg.solve(intr,uv)
            xy = (org+ray*((.045-org[2])/ray[2]))[:2]
            if np.abs(xy-NOMINAL).max()<.093:
                likely.append(xy)
    if not likely:
        raise ValueError("no overhead can silhouette in pickup area")
    def grid(center, radius, step):
        a = np.arange(-radius,radius+.1*step,step)
        return center+np.stack(np.meshgrid(a,a),-1).reshape(-1,2)
    def topfit(candidates):
        top,_ = rings(candidates,.09,cams[0])
        bot,_ = rings(candidates,0,cams[0])
        pts = np.rint(np.concatenate([top,bot],axis=1)).astype(np.int32)
        scores = []
        for p in pts:
            hull = cv2.convexHull(p).reshape(-1,2)
            a = np.maximum(hull.min(axis=0)-4,0)
            c = np.minimum(hull.max(axis=0)+5,[320,240])
            x,y = a
            xx,yy = c
            reg = dark[y:yy,x:xx]
            vis = ~yellow[y:yy,x:xx]
            mask = np.zeros(reg.shape,np.uint8)
            cv2.fillConvexPoly(mask,hull-a,1)
            score = np.sum(mask.astype(bool)&reg&vis)/max(1,np.sum((mask.astype(bool)|reg)&vis))
            scores.append(score)
        j = np.argmax(scores)
        return candidates[j],scores[j]
    top,_ = topfit(np.concatenate([grid(xy,.024,.004) for xy in likely]))
    top,quality = topfit(grid(top,.004,.001))
    if quality<.65:
        raise ValueError(f"overhead silhouette confidence {quality:.3f}")
    # Only match the visible portion of the projected top rim in the wrist image.
    edges = cv2.Canny(cv2.cvtColor(wrist,cv2.COLOR_RGB2GRAY),18,40)
    dt = cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    cand = grid(top,.008,.001)
    uv,z = rings(cand,.09,cams[1])
    px = np.rint(uv).astype(int)
    x,y = px[...,0],px[...,1]
    valid = (z>.01)&(x>2)&(x<317)&(y>2)&(y<237)
    ds = dt[np.clip(y,0,239),np.clip(x,0,319)]
    frac = (valid&(ds<2.5)).sum(1)/np.maximum(1,valid.sum(1))
    score = (np.minimum(ds,6)*valid).sum(1)/np.maximum(1,valid.sum(1)) + 20*np.linalg.norm(cand-top,axis=1)
    score[valid.sum(1)<22] = 1000
    j = np.argmin(score)
    use = bool(valid[j].sum()>=22 and frac[j]>=.65 and np.abs(cand[j]-top).max()<.0079)
    return top, (cand[j] if use else top), dict(overhead_iou=float(quality), wrist_used=use,
                                               wrist_visible=int(valid[j].sum()), wrist_inliers=float(frac[j]))
