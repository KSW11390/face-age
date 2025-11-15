# faceage/data/datasets.py
import os, glob, hashlib
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .transforms import build_transforms, to_pil

# UTKFace: race code
RACE_NAMES = ["White", "Black", "Asian", "Indian", "Others"]
NUM_RACES = len(RACE_NAMES)
AGE_GROUPS = [
    ("00-09", 0, 9),
    ("10-19", 10, 19),
    ("20-29", 20, 29),
    ("30-39", 30, 39),
    ("40-49", 40, 49),
    ("50-59", 50, 59),
    ("60-69", 60, 69),
    ("70-79", 70, 79),
    ("80-85", 80, 85),
]

def age_to_group_name(age: int) -> str:
    for name, lo, hi in AGE_GROUPS:
        if lo <= age <= hi:
            return name
    return "others"


def _parse_utkface_filename(path: str) -> Tuple[int, int, int]:
    base = os.path.basename(path)
    stem = base.split(".")[0]  # "23_0_2_201701161745"
    parts = stem.split("_")
    age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
    return age, gender, race

def _soft_label(age: int, num_bins: int = 86, sigma: float = 1.5) -> torch.Tensor:
    age = max(0, min(num_bins - 1, int(age)))
    xs = np.arange(num_bins, dtype=np.float32)
    g = np.exp(-0.5 * ((xs - age) / sigma) ** 2)
    g /= g.sum() + 1e-8
    return torch.from_numpy(g)

def _race_one_hot(race: int) -> torch.Tensor:
    idx = min(max(int(race), 0), NUM_RACES - 1)
    v = torch.zeros(NUM_RACES, dtype=torch.float32)
    v[idx] = 1.0
    return v

def _imread_rgb(path: str) -> np.ndarray:
    # PIL로 읽어서 RGB ndarray(HWC) 반환
    img = Image.open(path).convert("RGB")
    return np.array(img)

def _stable_split(paths: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    keys = []
    for p in paths:
        h = hashlib.md5((p + str(seed)).encode()).hexdigest()
        keys.append(int(h, 16))
    order = np.argsort(keys)
    n = len(paths)
    n_val = int(round(n * val_ratio))
    val_idx = set(order[:n_val])
    train, val = [], []
    for i, p in enumerate(paths):
        (val if i in val_idx else train).append(p)
    return train, val

# max_age(85)세 초과 샘플 제거
def _filter_by_age(paths: List[str], max_age: int = 85) -> List[str]:
    kept = []
    dropped = 0
    for p in paths:
        try:
            age, _, _ = _parse_utkface_filename(p)
            if age <= max_age:
                kept.append(p)
            else:
                dropped += 1
        except Exception:
            continue

    print(f"[UTKFace] Removed {dropped} images with age > {max_age}")
    assert len(kept) > 100, f"[UTKFace] Too few samples after filtering (<= {max_age}yrs)"
    return kept

@dataclass
class UTKFaceCfg:
    root: str
    split: str = "train"           # "train" | "val"
    img_size: int = 200
    num_bins: int = 86
    sigma: float = 1.5
    label_type: str = "soft" # "soft" | "hard"
    augment_minority_only: bool = False  # True면 White(0) 제외 인종만 train 증강

    aug_strength: str = "medium"   # "none" | "weak" | "medium" | "strong"
    use_age_group_aug: bool = False
    aug_dup: int = 1               # 한 이미지당 몇 배로 늘릴지

class UTKFaceDataset(Dataset):
    def __init__(self, cfg: UTKFaceCfg, file_list: Optional[List[str]] = None):
        self.cfg = cfg
        if file_list is None:
            all_files = sorted(glob.glob(os.path.join(cfg.root, "*.jpg")))
            if len(all_files) == 0:
                all_files = sorted(glob.glob(os.path.join(cfg.root, "*.png")))
            assert len(all_files) > 100, f"[UTKFace] No images under: {cfg.root}"
            self.paths = all_files
        else:
            self.paths = file_list

        # 🔥 배수 옵션 저장 (0 이하로 들어와도 최소 1)
        self.aug_dup = max(1, int(cfg.aug_dup))

        # 🔥 공통 train/eval transform
        #    - train 쪽은 cfg.aug_strength 반영
        self.tf_train = build_transforms(
            train=True,
            size=cfg.img_size,
            strength=cfg.aug_strength,
        )
        self.tf_eval  = build_transforms(
            train=False,
            size=cfg.img_size,
        )

        # 🔥 나이대별 증강 옵션이 켜져 있으면, 그룹별 transform dict 생성
        #    (0–19: strong, 20–49: medium, 50+: weak 예시)
        if cfg.use_age_group_aug:
            self.age_group_transforms = {}
            for name, lo, hi in AGE_GROUPS:
                if hi <= 19:          # 00–09, 10–19
                    strength = "medium"
                elif hi <= 39:        # 20–29, 30–39
                    strength = "weak"
                elif hi <= 59:        # 40–49, 50–59
                    strength = "medium"
                else:                 # 60–69, 70–79, 80–85
                    strength = "strong"

                self.age_group_transforms[name] = build_transforms(
                    train=True,
                    size=cfg.img_size,
                    strength=strength,
                )
        else:
            self.age_group_transforms = None

    def __len__(self) -> int:
        # 🔥 원본 개수 × 배수
        return len(self.paths) * self.aug_dup

    def __getitem__(self, idx: int):
        # 🔥 실제 파일 인덱스로 접기
        real_idx = idx // self.aug_dup
        path = self.paths[real_idx]

        try:
            age, gender, race = _parse_utkface_filename(path)
        except Exception:
            # 에러 나면 다음 샘플로 (dataset 길이 기준으로 순환)
            return self.__getitem__((idx + 1) % len(self))

        img_np = _imread_rgb(path)   # HWC RGB ndarray
        img_pil = to_pil(img_np)     # PIL.Image

        # ---------- 어떤 transform을 쓸지 결정 ----------
        if self.cfg.split == "train":
            # 1) 나이대별 증강이 켜져 있다면 → race 상관없이 age 기반으로 결정
            if self.age_group_transforms is not None:
                group_name = age_to_group_name(age)
                tf = self.age_group_transforms.get(group_name, self.tf_train)

            # 2) 아니면, 기존 augment_minority_only 규칙 사용
            else:
                if self.cfg.augment_minority_only:
                    tf = self.tf_train if race != 0 else self.tf_eval
                else:
                    tf = self.tf_train
        else:
            # val/test는 항상 eval transform
            tf = self.tf_eval

        img = tf(img_pil)

        # ---------- 라벨 생성 ----------
        if self.cfg.label_type == "soft":
            label = _soft_label(age, self.cfg.num_bins, self.cfg.sigma)
        else:
            label = torch.zeros(self.cfg.num_bins, dtype=torch.float32)
            label[min(int(age), self.cfg.num_bins - 1)] = 1.0

        race1h = _race_one_hot(race)
        return img.float(), label.float(), race1h, torch.tensor(age, dtype=torch.long)
# ---------- DataLoader Builder ----------

def _seed_worker(worker_id):
    import random
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)

def _env_defaults():
    # Colab(CUDA)면 workers/pin_memory 상승, macOS MPS/CPU면 안전값
    if torch.cuda.is_available():
        return dict(num_workers=2, pin_memory=True)
    else:
        return dict(num_workers=0, pin_memory=False)

def build_dataloaders(
    root: str,
    batch_size: int = 64,
    num_workers: Optional[int] = None,
    img_size: int = 200,
    num_bins: int = 86,
    sigma: float = 1.5,
    label_type: str = "soft",
    augment_minority_only: bool = False,
    val_ratio: float = 0.2,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    max_age: int = 85,

    # 🔥 새로 추가
    aug_strength: str = "medium",        # "none" | "weak" | "medium" | "strong"
    use_age_group_aug: bool = False,
    aug_dup: int = 1,

):
    all_paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
    if len(all_paths) == 0:
        all_paths = sorted(glob.glob(os.path.join(root, "*.png")))
    assert len(all_paths) > 100, f"[UTKFace] No images under: {root}"

    # 85세 초과 제거
    all_paths = _filter_by_age(all_paths, max_age=max_age)

    train_list, val_list = _stable_split(all_paths, val_ratio=val_ratio, seed=seed)

    train_ds = UTKFaceDataset(UTKFaceCfg(
        root=root,
        split="train",
        img_size=img_size,
        num_bins=num_bins,
        sigma=sigma,
        label_type=label_type,
        augment_minority_only=augment_minority_only,

        # 🔥 추가
        aug_strength=aug_strength,
        use_age_group_aug=use_age_group_aug,
        aug_dup=aug_dup,
    ), file_list=train_list)

    val_ds = UTKFaceDataset(UTKFaceCfg(
        root=root,
        split="val",
        img_size=img_size,
        num_bins=num_bins,
        sigma=sigma,
        label_type=label_type,
        augment_minority_only=False,

        # 🔥 val은 나이대별 증강도 배수도 쓰지 않는 게 일반적
        aug_strength="none",
        use_age_group_aug=False,
        aug_dup=1,
    ), file_list=val_list)
    if num_workers is None or pin_memory is None:
        d = _env_defaults()
        if num_workers is None: num_workers = d["num_workers"]
        if pin_memory is None: pin_memory = d["pin_memory"]

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=g, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=_seed_worker, generator=g, drop_last=False
    )
    return train_loader, val_loader