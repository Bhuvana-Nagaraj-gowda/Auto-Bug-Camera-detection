
import os, glob, json, cv2, numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

CLASSES = [
    "hdr", "demosaic", "rolling", "flare", "denoise", "banding"
]

class ArtifactDataset(Dataset):
    def __init__(self, root, size=384, split="train"):
        self.root = root
        self.size = size
        self.imgs = sorted(glob.glob(os.path.join(root, "images", "*.png")) + glob.glob(os.path.join(root, "images", "*.jpg")))
        self.masks = {c: os.path.join(root, "masks", c) for c in CLASSES}
        self.labels_json = os.path.join(root, "labels.json")
        with open(self.labels_json, "r") as f:
            self.labels = json.load(f)
        self.tfm = A.Compose([
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT),
            A.HorizontalFlip(p=0.5),
        ])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        p = self.imgs[idx]
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        name = os.path.splitext(os.path.basename(p))[0]
        masks = []
        cls = []
        for c in CLASSES:
            mp = os.path.join(self.masks[c], f"{name}.png")
            m = cv2.imread(mp, 0)
            if m is None:
                m = np.zeros(img.shape[:2], np.uint8)
                present = 0
            else:
                present = 1 if (m>0).any() else 0
            masks.append(m.astype(np.float32)/255.0)
            cls.append(present)
        masks = np.stack(masks, axis=0)
        aug = self.tfm(image=img, masks=[m for m in masks])
        img = aug["image"]
        masks = np.stack(aug["masks"], axis=0)

        img = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
        masks = torch.from_numpy(masks).float()
        cls = torch.tensor(cls).float()
        return img, masks, cls
