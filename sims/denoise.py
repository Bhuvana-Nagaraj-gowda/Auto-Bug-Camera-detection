
import numpy as np, cv2
from .utils import to_float, to_uint8

def simulate_denoise_smear(img, noise_sigma=0.08, smooth=7):
    imgf = to_float(img)
    noisy = np.clip(imgf + np.random.normal(0, noise_sigma, imgf.shape).astype(np.float32), 0, 1)
    # Over-smooth to create smear
    sm = cv2.GaussianBlur(noisy, (smooth, smooth), 0)
    mask = (np.abs(sm - noisy).mean(axis=2) > noise_sigma*0.6).astype(np.uint8)
    return to_uint8(sm), mask
