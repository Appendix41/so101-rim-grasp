"""Locate a flat charcoal patch and fit an inset support rectangle from calibrated RGB."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def table_points(uv, eye, rot, intr):
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(intr).T @ rot.T
    if np.any(rays[:,2] >= -.01):
        raise ValueError("table projection is behind camera")
    return (eye + rays * ((.0005-eye[2])/rays[:,2,None]))[:,:2]


def locate(rgb, eye, quat, intr, radius=.050, slack=.005):
    rot = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
    lum = rgb.max(2)
    mask = ((lum >= 55) & (lum <= 125) & (np.ptp(rgb,axis=2) <= 20)).astype("uint8")
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        pts = table_points(contour[:,0,:], eye, rot, intr).astype("float32")
        rect = cv2.minAreaRect(pts)
        sides = np.array(rect[1])
        if sides.min() < .115 or sides.max() > .184:
            continue
        fill = cv2.contourArea(pts) / np.prod(sides)
        if fill < .85 or sides.min()/2 <= radius+slack:
            continue
        center = np.array(rect[0])
        inset = (rect[0],tuple(sides-2*(radius+slack)),rect[2])
        candidates.append((center,cv2.boxPoints(rect),cv2.boxPoints(inset),float(fill)))
    if len(candidates)!=1:
        raise ValueError(f"{len(candidates)} unambiguous inset targets")
    return candidates[0]
