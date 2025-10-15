
import os, hydra, torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from data.dataset import ArtifactDataset, CLASSES
from models.autobugnet import UNetSmall
from train.losses import DiceLoss, BCEFocal

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = ArtifactDataset(cfg.data.root, size=cfg.data.size, split="train")
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    net = UNetSmall(in_ch=3, seg_ch=len(CLASSES)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.train.lr))
    seg_loss = DiceLoss()
    cls_loss = BCEFocal()

    os.makedirs("runs/checkpoints", exist_ok=True)

    for epoch in range(cfg.train.epochs):
        net.train()
        running=0.0
        for i,(img, seg, cls) in enumerate(dl):
            img, seg, cls = img.to(device), seg.to(device), cls.to(device)
            logits_seg, logits_cls = net(img)
            loss = seg_loss(logits_seg, seg) + 0.5*cls_loss(logits_cls, cls)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
            if (i+1)%20==0:
                print(f"Epoch {epoch+1} Iter {i+1}/{len(dl)} Loss {running/(i+1):.4f}")
        torch.save(net.state_dict(), f"runs/checkpoints/epoch_{epoch+1:03d}.pt")

    print("Training complete. Checkpoints in runs/checkpoints")

if __name__ == "__main__":
    main()
