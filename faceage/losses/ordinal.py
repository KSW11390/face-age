## option -> 필요 시 사용 ##

import torch
import torch.nn.functional as F

def coral_levels(age, num_bins):
    # levels[k] = 1 if age > k  (0..num_bins-2) 형태
    levels = torch.zeros(num_bins-1, dtype=torch.float32)
    levels[:max(age,0)] = 1.0
    return levels

def coral_loss(logits, levels):
    # logits: (B, K-1), levels: (B, K-1) with {0,1}
    log_p = torch.log(torch.sigmoid(logits) + 1e-8)
    log_1_p = torch.log(1 - torch.sigmoid(logits) + 1e-8)
    return -(levels * log_p + (1 - levels) * log_1_p).mean()