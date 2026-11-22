"""Visual-rule comparator from the frozen development-B study."""
import cv2
import numpy as np


def geometric_rule(proposal,alternatives,teacher,inset,restricted):
    if teacher is None:
        return proposal,"no_teacher_keep"
    def safe(x):
        return x is not None and cv2.pointPolygonTest(inset,tuple(map(float,x)),True)>=0
    if safe(proposal):
        return proposal,"inside_keep"
    if not restricted:
        return teacher,"expert_center"
    valid=[x for x in alternatives if safe(x)]
    return (min(valid,key=lambda x:np.linalg.norm(x-teacher)),"safe_menu") if valid else (None,"no_safe_option")
