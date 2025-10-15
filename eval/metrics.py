
import torch

def multilabel_accuracy(probs, targets):
    return ((probs>0.5) == (targets>0.5)).float().mean().item()
