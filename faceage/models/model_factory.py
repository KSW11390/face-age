# faceage/models/model_factory.py
from __future__ import annotations
from typing import Dict, Callable, Optional
import torch.nn as nn

# =========================================================
# - Input: (B, C, H, W)
# - Output: (B, feat_dim)
# - DownSample: 5회 → 200→100→50→25→12→6
# - AdaptiveAvgPool2d((1,1))
# =========================================================

# ---------------- VGG-style ----------------
class VGGStyle(nn.Module):
    """
    5개 스테이지, 각 스테이지마다: Conv-BN-Act-Conv-BN-Act-MaxPool(2)
    공간 크기: 200 → 100 → 50 → 25 → 12 → 6
    """
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 128,
        width: int = 64,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        act = activation or nn.ReLU()
        dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        stages = []
        in_ch = in_channels
        ch = width
        for _ in range(5):  # 5번 다운샘플
            stages += [
                nn.Conv2d(in_ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch), act,
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch), act,
                dp,
                nn.MaxPool2d(kernel_size=2, stride=2),  # /2
            ]
            in_ch = ch
            ch *= 2  # 다음 스테이지 채널 2배

        self.features = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(in_ch, feat_dim)  # in_ch는 마지막 스테이지 채널

    def forward(self, x):
        x = self.features(x)              # (B, C, 6, 6)
        x = self.pool(x).flatten(1)       # (B, C)
        x = self.proj(x)                  # (B, feat_dim)
        return x


# ---------------- ResNet-style ----------------
class BasicBlockExactHalf(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        act = activation or nn.ReLU()
        self.act = act
        self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if stride == 2:
            # 정확히 /2
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.down = None
        if stride == 2 or in_ch != out_ch:
            if stride == 2:
                self.down = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            else:
                self.down = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.dp(out)
        out = self.act(out + identity)
        return out


class ResNetStyle(nn.Module):
    """
    다운샘플 총 5회:
    - stem: conv7x7(s=2) → MaxPool2d(2)  → 200→100→50 (2회)
    - stage1: stride=1 (크기 유지)
    - stage2: stride=2 → 25 (3회)
    - stage3: stride=2 → 12 (4회, BasicBlockExactHalf가 정확히 25→12)
    - stage4: stride=2 → 6  (5회)
    """
    def __init__(
        self,
        in_channels: int = 3,
        feat_dim: int = 128,
        width: int = 64,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        act = activation or nn.ReLU()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        chs = [width, width*2, width*4, width*8]
        self.stage1 = nn.Sequential(
            BasicBlockExactHalf(chs[0], chs[0], stride=1, activation=act, dropout=dropout),
            BasicBlockExactHalf(chs[0], chs[0], stride=1, activation=act, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            BasicBlockExactHalf(chs[0], chs[1], stride=2, activation=act, dropout=dropout),  # 50->25
            BasicBlockExactHalf(chs[1], chs[1], stride=1, activation=act, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            BasicBlockExactHalf(chs[1], chs[2], stride=2, activation=act, dropout=dropout),  # 25->12
            BasicBlockExactHalf(chs[2], chs[2], stride=1, activation=act, dropout=dropout),
        )
        self.stage4 = nn.Sequential(
            BasicBlockExactHalf(chs[2], chs[3], stride=2, activation=act, dropout=dropout),  # 12->6
            BasicBlockExactHalf(chs[3], chs[3], stride=1, activation=act, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(chs[3], feat_dim)

    def forward(self, x):
        x = self.stem(x)      # 200->100->50
        x = self.stage1(x)    # 50
        x = self.stage2(x)    # 25
        x = self.stage3(x)    # 12
        x = self.stage4(x)    # 6
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        return x

REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "vgg": VGGStyle,
    "resnet": ResNetStyle,
}

def build_model(
    kind: str,
    in_channels: int = 3,
    feat_dim: int = 128,
    width: int = 64,
    activation: Optional[nn.Module] = None,
    dropout: float = 0.0,
) -> nn.Module:
    if kind not in REGISTRY:
        raise ValueError(f"Unknown model_type: {kind}. choices={list(REGISTRY.keys())}")
    return REGISTRY[kind](
        in_channels=in_channels,
        feat_dim=feat_dim,
        width=width,
        activation=activation,
        dropout=dropout,
    )