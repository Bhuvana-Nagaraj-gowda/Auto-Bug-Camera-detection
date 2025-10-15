
import torch, torch.nn as nn

class BCEFocal(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        pt = torch.exp(-bce)
        loss = self.alpha * (1-pt)**self.gamma * bce
        if self.reduction=="mean": return loss.mean()
        if self.reduction=="sum": return loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2*(probs*targets).sum(dim=(2,3))
        den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.eps
        loss = 1 - (num/den).mean()
        return loss
