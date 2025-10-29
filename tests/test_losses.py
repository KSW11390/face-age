import torch
from faceage.losses.soft_label import soft_ce_loss

def test_soft_ce_runs():
    logits = torch.randn(8, 86)
    soft = torch.softmax(torch.randn(8, 86), dim=-1)
    loss = soft_ce_loss(logits, soft)
    assert loss.dim()==0 and loss.item()>=0