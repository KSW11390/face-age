# faceage/data/transforms.py
from PIL import Image
import torchvision.transforms as T

def build_transforms(train: bool, size: int = 200):
    """
    torchvision 기반 최소/안정 증강.
    - PIL.Image 입력 → torch.Tensor 출력
    """
    if train:
        tf = T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    else:
        tf = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    return tf

def to_pil(img_np):
    """HWC RGB ndarray -> PIL.Image"""
    return Image.fromarray(img_np)