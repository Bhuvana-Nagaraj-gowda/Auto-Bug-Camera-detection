
import numpy as np, cv2
from .utils import to_float, to_uint8

def simulate_hdr_halo_ghost(img, strength=0.6, shift=(2,0)):
    """
    Create HDR halos/ghosting by fusing misaligned exposure pairs and
    adding tone-mapping overshoot around edges.
    Returns: (aug_img, mask) where mask is 1 for artifact regions.
    """
    imgf = to_float(img)
    # Make a brightened and darkened version
    bright = np.clip(imgf * (1.0 + strength), 0, 1)
    dark   = np.clip(imgf * (1.0 - 0.5*strength), 0, 1)

    # Misalign the bright frame to induce ghosting
    M = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
    bright_shift = cv2.warpAffine(bright, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Simple fusion + local edge overshoot
    fused = 0.6*imgf + 0.4*bright_shift
    # Overshoot: sharpen then clip
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
    sharp = cv2.filter2D(fused, -1, k)
    overshoot = np.clip(sharp, 0, 1)

    # Artifact mask: where misalignment changed edges
    edges = cv2.Canny((imgf*255).astype(np.uint8), 60, 120)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    diff = (np.abs(overshoot - fused).mean(axis=2) > 0.05).astype(np.uint8)*255
    mask = cv2.bitwise_and(edges, diff)
    mask = (cv2.GaussianBlur(mask, (0,0), 1.0) > 16).astype(np.uint8)

    return to_uint8(overshoot), mask
