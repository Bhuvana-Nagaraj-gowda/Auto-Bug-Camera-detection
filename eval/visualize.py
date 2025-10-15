
import cv2, numpy as np

def overlay_heatmap(img, mask, alpha=0.5):
    if mask.ndim==2:
        mask = mask[...,None]
    heat = (mask*255).astype(np.uint8)
    heat = heat.max(axis=2) if heat.shape[2]>1 else heat[...,0]
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 1.0, heat, alpha, 0)
    return out
