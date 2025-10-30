# faceage/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_transforms(train: bool, size: int = 200):
    """
    최소 증강 버전 (macOS MPS에서도 안전)
    train=True  -> Flip + 약간의 밝기조절
    train=False -> 리사이즈만
    """
    if train:
        tf = A.Compose([
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        tf = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    return tf