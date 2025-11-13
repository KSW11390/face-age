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
def build_transforms(train: bool, size: int = 200):
    normalize = T.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))

    if train:
        tf = T.Compose([
            T.Pad(padding=int(size * 0.08), padding_mode="reflect"),
            T.RandomCrop(size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=12, translate=(0.06, 0.06),
                           scale=(0.9, 1.1), shear=(-6, 6)),
            T.RandomPerspective(distortion_scale=0.20, p=0.30),
            T.ColorJitter(brightness=0.25, contrast=0.25,
                          saturation=0.20, hue=0.03),
            T.RandomGrayscale(p=0.05),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
            normalize,
            T.RandomErasing(p=0.25, scale=(0.02, 0.12),
                            ratio=(0.3, 3.3), value="random"),
        ])
    else:
        tf = T.Compose([
            T.Resize(int(size * 1.12)),
            T.CenterCrop(size),
            T.ToTensor(),
            normalize,
        ])
    return tf

# to_pil
# numpy 배열 (HWC,RGB) -> PIL.image 객체로 변환
def to_pil(img_np):
    return Image.fromarray(img_np)