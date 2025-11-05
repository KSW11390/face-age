import torch
import torch.nn as nn


class SoftHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_bins: int,
        race_dim: int = 5,
        dropout: float = 0.0,
        use_race=True,
    ):
        super().__init__()
        self.use_race = use_race
        merged = in_dim + (race_dim if use_race else 0)
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(merged, num_bins))

    def forward(self, feat, race_onehot=None):
        if self.use_race and race_onehot is not None:
            feat = torch.cat([feat, race_onehot], dim=1)
        return self.net(feat)  # logits
