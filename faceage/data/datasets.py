# faceage/data/datasets.py
import os, glob, hashlib
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import build_transforms

# UTKFace: race code (dataset 정의와 동일)
RACE_NAMES = ["White", "Black", "Asian", "Indian", "Others"]
NUM_RACES = len(RACE_NAMES)


def _parse_utkface_filename(path: str) -> Tuple[int, int, int]:
    """
    path .../23_0_2_201701161745.jpg.chip.jpg  -> (age=23, gender=0, race=2)
    간혹 예외 파일은 try/except로 스킵.
    """
    base = os.path.basename(path)
    stem = base.split(".")[0]  # "23_0_2_201701161745"
    parts = stem.split("_")
    age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
    return age, gender, race


def _soft_label(age: int, num_bins: int = 86, sigma: float = 1.5) -> torch.Tensor:
    """가우시안 기반 soft-classification 타깃 (0~85세 가정)."""
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
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1]  # BGR -> RGB


def _stable_split(paths: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    파일 경로 해시 기반으로 재현성 있는 train/val 분할.
    - 팀원/세션이 달라도 항상 동일 split.
    """
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


@dataclass
class UTKFaceCfg:
    root: str
    split: str = "train"           # "train" | "val"
    img_size: int = 200
    num_bins: int = 86
    sigma: float = 1.5
    augment_minority_only: bool = False  # True면 White(0) 제외 인종만 train 증강


class UTKFaceDataset(Dataset):
    def __init__(self, cfg: UTKFaceCfg, file_list: Optional[List[str]] = None):
        self.cfg = cfg
        if file_list is None:
            all_files = sorted(glob.glob(os.path.join(cfg.root, "*.jpg")))
            if len(all_files) == 0:
                all_files = sorted(glob.glob(os.path.join(cfg.root, "*.png")))
            assert len(all_files) > 100, f"[UTKFace] No images under: {cfg.root}"
            # 내부에서 분할은 build_dataloaders에서 수행(여긴 그대로 받음)
            self.paths = all_files
        else:
            self.paths = file_list

        # transforms — 간소화 버전
        self.tf_train = build_transforms(train=True, size=cfg.img_size)
        self.tf_eval  = build_transforms(train=False, size=cfg.img_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            age, gender, race = _parse_utkface_filename(path)
        except Exception:
            # 드문 예외 파일은 건너뛰고 다음 샘플로 대체
            return self.__getitem__((idx + 1) % len(self.paths))

        img = _imread_rgb(path)

        # 증강 규칙
        if self.cfg.split == "train":
            if self.cfg.augment_minority_only and race != 0:  # 0=White
                img = self.tf_train(image=img)["image"]
            else:
                img = self.tf_eval(image=img)["image"]
        else:
            img = self.tf_eval(image=img)["image"]

        soft = _soft_label(age, self.cfg.num_bins, self.cfg.sigma)  # (num_bins,)
        race1h = _race_one_hot(race)                                 # (5,)

        return img.float(), soft.float(), race1h, torch.tensor(age, dtype=torch.long)


# ---------- DataLoader Builder ----------

def _seed_worker(worker_id):
    # 각 DataLoader worker의 시드 고정
    import random
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)


def build_dataloaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 0,  # macOS MPS 환경에서는 0이 안전
    img_size: int = 200,
    num_bins: int = 86,
    sigma: float = 1.5,
    augment_minority_only: bool = False,
    val_ratio: float = 0.1,
    seed: int = 42,
    pin_memory: bool = False,
):
    """
    root: UTKFace 이미지가 들어있는 디렉토리
    augment_minority_only: White(0) 제외 인종만 train 증강
    """
    all_paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
    if len(all_paths) == 0:
        all_paths = sorted(glob.glob(os.path.join(root, "*.png")))
    assert len(all_paths) > 100, f"[UTKFace] No images under: {root}"

    train_list, val_list = _stable_split(all_paths, val_ratio=val_ratio, seed=seed)

    train_ds = UTKFaceDataset(UTKFaceCfg(
        root=root, split="train", img_size=img_size, num_bins=num_bins, sigma=sigma,
        augment_minority_only=augment_minority_only
    ), file_list=train_list)

    val_ds = UTKFaceDataset(UTKFaceCfg(
        root=root, split="val", img_size=img_size, num_bins=num_bins, sigma=sigma,
        augment_minority_only=False
    ), file_list=val_list)

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