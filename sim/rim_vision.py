"""Recover the ground-plane centre of an upright can in the calibrated sim view."""

import numpy as np
from scipy.ndimage import label


def centre_xy(rgb, radius):
    if rgb.shape != (240, 320, 3):
        raise ValueError("Expected the calibrated overhead RGB view")
    mask = np.max(rgb, axis=2) < 65
    cc, n = label(mask)
    if not n:
        raise ValueError("No visible can")
    areas = np.bincount(cc.ravel())[1:]
    ys, xs = np.where(cc == areas.argmax()+1)
    if not (500 <= len(xs) <= 6000 and 25 <= np.ptp(xs) <= 75 and 15 <= np.ptp(ys) <= 100):
        raise ValueError("No plausible isolated can silhouette")
    v = ys.max()+.5
    u = xs[ys >= v-2].mean()
    eye = np.array([.18, -.58, .52])
    forward = np.array([.025, -.21, .07]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    focal_px = 320*22/28
    ray = forward + right*(u-159.5)/focal_px - up*(v-119.5)/focal_px
    contact = (eye - eye[2]/ray[2]*ray)[:2]
    radial = contact-eye[:2]
    return contact + radius * radial/np.linalg.norm(radial)
