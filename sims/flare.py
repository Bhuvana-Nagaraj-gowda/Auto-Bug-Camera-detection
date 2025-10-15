
import numpy as np, cv2
from .utils import to_float, to_uint8

def simulate_flare(img, strength=0.6, radius=80):
    imgf = to_float(img)
    h,w,_ = img.shape
    cx, cy = int(0.1*w), int(0.2*h)
    Y, X = np.ogrid[:h,:w]
    d = np.sqrt((X-cx)**2+(Y-cy)**2)
    glare = np.exp(-(d**2)/(2*(radius**2))) * strength
    veil = 0.25*strength
    out = np.clip(imgf + glare[...,None] + veil, 0, 1)
    mask = (glare > 0.1).astype(np.uint8)
    return to_uint8(out), mask
