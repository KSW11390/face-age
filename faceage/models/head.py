# faceage/models/head.py
import torch
import torch.nn as nn

NUM_RACES = 5 # (White, Black, Asian, Indian, Others)

class SoftHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_bins: int,
        use_race: bool = False,
        race_dim: int = NUM_RACES,
        hidden: int = 256,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.use_race = use_race
        in_dim = feat_dim + (race_dim if use_race else 0)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            activation,
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, num_bins, bias=True),
        )

    def forward(self, feats: torch.Tensor, races: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_race:
            if races is None:
                raise ValueError("SoftHead(use_race=True)인데 races=None이 들어왔습니다.")
            x = torch.cat([feats, races], dim=1)
        else:
            x = feats
        return self.net(x)