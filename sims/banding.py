
import numpy as np, cv2
from .utils import to_float, to_uint8

def simulate_banding(img, period=16, depth=32):
    imgf = to_float(img)
    # quantize
    q = np.floor(imgf * depth) / max(depth-1,1)
    # add horizontal bands
    h, w, _ = img.shape
    y = np.arange(h)[:,None]
    band = (0.08*np.sin(2*np.pi*y/period)).astype(np.float32)
    out = np.clip(q + band, 0, 1)
    mask = (np.abs(out - imgf).mean(axis=2) > 0.04).astype(np.uint8)
    return to_uint8(out), mask
