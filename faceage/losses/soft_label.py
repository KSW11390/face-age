import torch.nn.functional as F

def soft_ce_loss(logits, soft_targets):
    logprob = F.log_softmax(logits, dim=-1)
    return -(soft_targets * logprob).sum(dim=-1).mean()