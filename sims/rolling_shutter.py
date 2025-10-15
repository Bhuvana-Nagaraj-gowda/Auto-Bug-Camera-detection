
import numpy as np, cv2
from .utils import to_float, to_uint8

def simulate_rolling_shutter(img, skew=0.15):
    """
    Shears rows progressively to mimic rolling-shutter skew on moving subjects.
    """
    h, w, _ = img.shape
    imgf = to_float(img)
    out = np.zeros_like(imgf)
    mask = np.zeros((h,w), np.uint8)
    for r in range(h):
        shift = int(skew * (r/h) * w * 0.5)
        out[r] = np.roll(imgf[r], shift, axis=0)
        if abs(shift) > 0:
            mask[r, max(0,min(w-1, shift))] = 1
    import cv2
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), 2)
    return to_uint8(out), mask
