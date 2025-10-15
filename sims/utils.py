
import numpy as np
import cv2

def to_uint8(img):
    img = np.clip(img, 0, 1)
    return (img * 255.0 + 0.5).astype(np.uint8)

def to_float(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)
