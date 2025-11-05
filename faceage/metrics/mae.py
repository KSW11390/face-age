import torch


def expected_age_from_logits(logits):
    prob = torch.softmax(logits, dim=-1)
    bins = torch.arange(prob.size(1), device=prob.device, dtype=prob.dtype)
    return (prob * bins).sum(dim=1)


def mae(pred_age, true_age):
    return (pred_age.float() - true_age.float()).abs().mean().item()
