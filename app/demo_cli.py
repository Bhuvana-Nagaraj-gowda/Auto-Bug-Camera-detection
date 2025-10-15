
import os, glob, argparse, cv2, torch
import numpy as np
from models.autobugnet import UNetSmall
from data.dataset import CLASSES
from eval.visualize import overlay_heatmap

def load_net(weights=None, device="cpu"):
    net = UNetSmall(in_ch=3, seg_ch=len(CLASSES)).to(device)
    if weights and os.path.isfile(weights):
        net.load_state_dict(torch.load(weights, map_location=device))
    net.eval(); return net

def infer_folder(images, out, weights=None, size=384, device="cpu"):
    os.makedirs(out, exist_ok=True)
    net = load_net(weights, device)
    for p in sorted(glob.glob(os.path.join(images, "*.*"))):
        img0 = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        h,w = img0.shape[:2]
        img = cv2.resize(img0, (size,size))
        ten = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)/255.0
        ten = ten.to(device)
        with torch.no_grad():
            seg, cls = net(ten)
            seg = torch.sigmoid(seg)[0].cpu().numpy().transpose(1,2,0)
        seg = cv2.resize(seg, (w,h))
        overlay = overlay_heatmap(cv2.cvtColor(img0, cv2.COLOR_RGB2BGR), seg, alpha=0.5)
        name = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(out, f"{name}_overlay.png"), overlay)
    print(f"Saved overlays to {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    infer_folder(args.images, args.out, args.weights, args.size, args.device)
