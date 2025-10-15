
import os, json, cv2, argparse, numpy as np
from sims.hdr import simulate_hdr_halo_ghost
from sims.demosaic import simulate_demosaic_zipper_moire
from sims.rolling_shutter import simulate_rolling_shutter
from sims.flare import simulate_flare
from sims.denoise import simulate_denoise_smear
from sims.banding import simulate_banding

CLASSES = ["hdr","demosaic","rolling","flare","denoise","banding"]

def ensure_dirs(out):
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(out, "masks", c), exist_ok=True)

def random_image(w=512, h=384):
    # Generate a simple synthetic scene (rectangles, circles, text) to exercise edges/frequencies
    img = np.ones((h,w,3), np.uint8)*np.random.randint(150, 230)
    for _ in range(np.random.randint(5, 12)):
        color = np.random.randint(0,255,(3,), dtype=np.uint8)
        if np.random.rand() < 0.5:
            p1 = (np.random.randint(0,w), np.random.randint(0,h))
            p2 = (np.random.randint(0,w), np.random.randint(0,h))
            cv2.rectangle(img, p1, p2, color.tolist(), -1)
        else:
            c = (np.random.randint(0,w), np.random.randint(0,h))
            r = np.random.randint(10, min(w,h)//4)
            cv2.circle(img, c, r, color.tolist(), -1)
        if np.random.rand() < 0.4:
            cv2.putText(img, "AutoBugCam", (np.random.randint(0,w-120), np.random.randint(20,h-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=200)
    args = ap.parse_args()

    ensure_dirs(args.out)
    labels = {}
    for i in range(args.num):
        base = f"img_{i:05d}"
        img = random_image()
        masks = {c: np.zeros(img.shape[:2], np.uint8) for c in CLASSES}
        # Randomly sample which artifacts to apply (1-3)
        k = np.random.randint(1, 4)
        choices = np.random.choice(CLASSES, size=k, replace=False)
        aug = img.copy()
        for c in choices:
            if c=="hdr":
                aug, m = simulate_hdr_halo_ghost(aug, strength=np.random.uniform(0.4,0.9), shift=(np.random.randint(-3,3), np.random.randint(-2,2)))
            if c=="demosaic":
                aug, m = simulate_demosaic_zipper_moire(aug, strength=np.random.uniform(0.3,0.9))
            if c=="rolling":
                aug, m = simulate_rolling_shutter(aug, skew=np.random.uniform(-0.2,0.2))
            if c=="flare":
                aug, m = simulate_flare(aug, strength=np.random.uniform(0.3,0.9), radius=np.random.randint(40,120))
            if c=="denoise":
                aug, m = simulate_denoise_smear(aug, noise_sigma=np.random.uniform(0.05,0.12), smooth=np.random.choice([5,7,9]))
            if c=="banding":
                aug, m = simulate_banding(aug, period=np.random.choice([8,12,16,24]), depth=np.random.choice([16,32,64]))
            masks[c] = np.maximum(masks[c], (m>0).astype(np.uint8))

        cv2.imwrite(os.path.join(args.out, "images", base + ".png"), aug)
        for c in CLASSES:
            cv2.imwrite(os.path.join(args.out, "masks", c, base + ".png"), (masks[c]*255).astype(np.uint8))
        labels[base] = {c: int(c in choices) for c in CLASSES}

    with open(os.path.join(args.out, "labels.json"), "w") as f:
        import json
        json.dump(labels, f, indent=2)

    print(f"Generated {args.num} images at {args.out}")

if __name__ == "__main__":
    main()
