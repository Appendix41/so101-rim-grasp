"""Frozen RGB proposal, teacher, and fixed carry functions from the simulator."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from cloth_safe_target import locate, table_points
from placement_scene import adaptive_grasp_z


def corroborated_view(images,calibration):
    centers=[]
    for img in images:
        try:
            centers.append(locate(img,*calibration)[0])
        except ValueError:
            centers.append(np.full(2,np.nan))
    for j in range(len(centers)):
        if any(np.linalg.norm(centers[j]-x)<=.008 for x in centers[:j]):
            return j,np.stack(centers)
    return None,np.stack(centers)


def anonymous_regions(images,cal):
    eye,q,K=cal
    rot=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    frames=[]
    for img in images:
        a=img.astype(np.int16)
        chroma=a.max(2)-a.min(2)
        mask=((chroma>60)|((a.max(2)>=55)&(a.max(2)<=125)&(chroma<=20))).astype("uint8")
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        found=[]
        for c in cs:
            if cv2.contourArea(c)<500:
                continue
            pts=table_points(c[:,0,:],eye,rot,K).astype("float32")
            rc=cv2.minAreaRect(pts)
            dims=np.array(rc[1])
            if dims.min()>=.115 and dims.max()<=.184 and cv2.contourArea(pts)/dims.prod()>=.81:
                found.append(np.array(rc[0]))
        frames.append(found)
    unique=[]
    for j,frame in enumerate(frames[1:],1):
        for xy in frame:
            if any(np.linalg.norm(xy-z)<=.008 for fr in frames[:j] for z in fr):
                if all(np.linalg.norm(xy-z)>.012 for z in unique):
                    unique.append(xy)
    return sorted(unique,key=lambda x:tuple(x))


def visual_label(images,cal,proposal):
    pick,_=corroborated_view(images,cal)
    if pick is None:
        return None,None,"teacher_unavailable"
    target,_,inset,_=locate(images[pick],*cal)
    if proposal is None:
        reason="inconsistent_student"
    else:
        slack=cv2.pointPolygonTest(inset,tuple(np.asarray(proposal,dtype=float)),True)
        reason="inside_safe_region" if slack>=0 else "outside_safe_region"
    return target,inset,reason


def carry(seg,xy,solve,home,jaw):
    dest=xy+[.049,-.008]
    z=adaptive_grasp_z(90)
    up=solve(np.r_[dest,z+.065],seg.q,jaw)
    down=solve(np.r_[dest,z],up,jaw)
    opened=down.copy()
    opened[-1]=home[-1]
    retreat=solve(np.r_[dest,z+.065],down,home[-1])
    for q,n,phase in ((up,36,"transfer"),(down,20,"lower"),(opened,8,"release"),
                      (retreat,18,"retreat"),(home,18,"clear"),(home,30,"settle")):
        seg.move(q,n,phase)
