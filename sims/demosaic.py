
import numpy as np, cv2
from .utils import to_float, to_uint8

def mosaic_bayer(img):
    h, w, _ = img.shape
    out = np.zeros_like(img)
    # RGGB pattern
    out[0::2,0::2,1] = img[0::2,0::2,1]  # G
    out[0::2,1::2,2] = img[0::2,1::2,2]  # B
    out[1::2,0::2,0] = img[1::2,0::2,0]  # R
    out[1::2,1::2,1] = img[1::2,1::2,1]  # G
    return out

def poor_demosaic(mosaicked):
    # naive bilinear demosaic to intentionally create zippering/moire
    return cv2.cvtColor(mosaicked, cv2.COLOR_BGR2GRAY)

def simulate_demosaic_zipper_moire(img, strength=0.6):
    imgf = to_float(img)
    # add high freq pattern
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h,0:w]
    freq = 10 + int(10*strength)
    pattern = (np.sin(2*np.pi*xx/freq)+1)/2
    patterned = np.clip(imgf*(0.7+0.6*pattern[...,None]), 0, 1)

    mosa = mosaic_bayer(patterned)
    # Upsample grayscale demosaic to 3ch to simulate color loss + zippering
    gray = poor_demosaic((mosa*255).astype(np.uint8)).astype(np.float32)/255.0
    demosa = np.stack([gray, gray, gray], axis=2)

    # Artifact mask where local contrast flips
    lap = cv2.Laplacian((demosa*255).astype(np.uint8), cv2.CV_16S, ksize=3)
    lap = np.abs(lap).astype(np.uint8)
    mask = (lap.mean(axis=2) > 12).astype(np.uint8)

    return to_uint8(demosa), mask
