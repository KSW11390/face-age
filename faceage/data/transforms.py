from PIL import Image
import torchvision.transforms as T


def build_transforms(
    train: bool,
    size: int = 200,
    strength: str = "medium",
    use_random_erase: bool = False,
    erase_prob: float = 0.2,
):
    normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    # train set이 아닐 경우 바로 return
    if not train:
        return T.Compose(
            [
                T.Resize(size),
                T.CenterCrop(size),
                T.ToTensor(),
                normalize,
            ]
        )

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
            T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.97, 1.03)),
        ]
    elif strength == "medium":
        aug = [
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.RandomPerspective(distortion_scale=0.2, p=0.4),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    elif strength == "strong":  # randomgrayscale, gausianblurr, erase
        aug = [
            T.RandomAffine(degrees=18, translate=(0.08, 0.08), scale=(0.85, 1.2)),
            T.RandomPerspective(distortion_scale=0.35, p=0.6),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
            T.RandomApply([T.RandomGrayscale(p=0.2)], p=0.3),
        ]
    else:
        raise ValueError(f"Unknown aug strength: {strength}")

    tail = [
        T.ToTensor(),
        normalize,
    ]

    if use_random_erase and strength != "none":
        tail.append(T.RandomErasing(p=erase_prob, scale=(0.02, 0.15), ratio=(0.3, 3.3)))

    return T.Compose(base + aug + tail)


# to_pil
# numpy 배열 (HWC,RGB) -> PIL.image 객체로 변환
def to_pil(img_np):
    return Image.fromarray(img_np)
