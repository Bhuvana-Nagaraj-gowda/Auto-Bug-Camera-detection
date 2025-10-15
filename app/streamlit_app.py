
import os
import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import torch
import cv2

# Allow imports from project root when running `streamlit run app/streamlit_app.py`
ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.autobugnet import UNetSmall
from data.dataset import CLASSES
from eval.visualize import overlay_heatmap

st.set_page_config(page_title="AutoBugCam Dashboard", layout="wide")

st.title("AutoBugCam — Artifact Heatmap Dashboard")
st.write("Upload images, visualize per-pixel artifact heatmaps, and save overlays.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    device = "cuda" if torch.cuda.is_available() and st.checkbox("Use CUDA if available", value=False) else "cpu"
    size = st.slider("Inference size (square resize)", min_value=256, max_value=768, value=384, step=32)
    alpha = st.slider("Heatmap alpha", 0.1, 0.9, 0.5, 0.05)
    weight_path = st.text_input("Optional weights (.pt)", value="")
    save_dir = st.text_input("Save outputs to", value="runs/streamlit")
    os.makedirs(save_dir, exist_ok=True)

    st.markdown("---")
    st.caption("Need sample data? Generate via:")
    st.code("python data/generate_synth.py --out data/synth --num 200", language="bash")

@st.cache_resource
def load_net(weights=None, device="cpu"):
    net = UNetSmall(in_ch=3, seg_ch=len(CLASSES)).to(device)
    if weights and os.path.isfile(weights):
        state = torch.load(weights, map_location=device)
        net.load_state_dict(state)
    net.eval()
    return net

def infer_image(pil_img, net, size=384, device="cpu"):
    img0 = np.array(pil_img.convert("RGB"))
    h,w = img0.shape[:2]
    img = cv2.resize(img0, (size, size))
    ten = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)/255.0
    ten = ten.to(device)
    with torch.no_grad():
        seg, cls = net(ten)
        seg = torch.sigmoid(seg)[0].cpu().numpy().transpose(1,2,0)
        cls = torch.sigmoid(cls)[0].cpu().numpy()
    seg = cv2.resize(seg, (w, h))
    overlay = overlay_heatmap(cv2.cvtColor(img0, cv2.COLOR_RGB2BGR), seg, alpha=alpha)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return img0, seg, cls, overlay

# Load model
net = load_net(weights=weight_path if weight_path.strip() else None, device=device)

# File uploader
uploaded = st.file_uploader("Upload one or more images", type=["png","jpg","jpeg"], accept_multiple_files=True)

if uploaded:
    cols = st.columns(2)
    for file in uploaded:
        pil = Image.open(file)
        with st.spinner(f"Analyzing {file.name} ..."):
            img0, seg, cls, overlay = infer_image(pil, net, size=size, device=device)
        # Display original and overlay
        cols[0].subheader(file.name)
        cols[0].image(img0, caption="Original", use_column_width=True)
        cols[1].subheader("Overlay")
        cols[1].image(overlay, caption="Artifact Heatmap", use_column_width=True)

        # Show per-class scores
        scores = {c: float(cls[i]) for i, c in enumerate(CLASSES)}
        st.markdown("**Per-class scores (0-1):**")
        st.json(scores)

        # Save button
        out_path = os.path.join(save_dir, Path(file.name).stem + "_overlay.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        st.success(f"Saved to: {out_path}")

st.markdown("---")
st.caption("Tip: Train your own weights, point the 'weights' path to a .pt checkpoint, and re-run inference.")
