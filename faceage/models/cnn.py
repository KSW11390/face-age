import torch.nn as nn


class SimpleCNN(nn.Module):  #
    def __init__(self, in_channels=3, feat_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Batch, 3, 200, 200
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Batch, 64, 100, 100
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Batch, 128, 50, 50
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 전역 평균 풀링
        self.proj = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)  # (B, 128)
        return x
