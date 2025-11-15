from PIL import Image
import torchvision.transforms as T

# build_transforms
# train/val 여부에 따라 (augmentation) + preprocessing 이후 torchvision.transforms.Compose를 Return
# Parameters
#   - train
#       - True: augmentation + preprocessing
#       - False: preprocessing
#   - size: 최종으로 맞춰줄 해상도
# 
def build_transforms(train: bool, size: int = 200, strength: str = "medium"):

    normalize = T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
    # train set이 아닐 경우 바로 return
    if not train:
        return T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            normalize,
        ])
    
    # 공통 전처리 (train set)
    base = [
        T.Pad(padding=int(size * 0.08), padding_mode="reflect"),
        T.RandomCrop(size),
        T.RandomHorizontalFlip(p=0.5),
    ]

    if strength == "none":
        aug = []
    elif strength == "weak":
        aug = [
            T.RandomAffine(
                degrees=8,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                shear=(-4, 4),
            ),
        ]
    elif strength == "medium":
        aug = [
            T.RandomAffine(
                degrees=12,
                translate=(0.06, 0.06),
                scale=(0.9, 1.1),
                shear=(-6, 6),
            ),
            T.RandomPerspective(distortion_scale=0.15, p=0.3),
        ]
    elif strength == "strong": # randomgrayscale, gausianblurr, erase
        aug = [
            T.RandomAffine(
                degrees=15,
                translate=(0.08, 0.08),
                scale=(0.85, 1.15),
                shear=(-8, 8),
            ),
            T.RandomPerspective(distortion_scale=0.25, p=0.5),
            T.ColorJitter(brightness=0.25, contrast=0.25),
        ]
    else:
        raise ValueError(f"Unknown aug strength: {strength}")
    
    tail = [
        T.ToTensor(),
        normalize,
        T.RandomErasing(p=0.2 if strength != "none" else 0.0,
                        scale=(0.02, 0.15),
                        ratio=(0.3, 3.3)),
    ]

    return T.Compose(base + aug + tail)

# to_pil
# numpy 배열 (HWC,RGB) -> PIL.image 객체로 변환
def to_pil(img_np):
    return Image.fromarray(img_np)