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

@dataclass
class UTKFaceCfg:
    root: str
    split: str = "train"           # "train" | "val"
    img_size: int = 200
    num_bins: int = 86
    sigma: float = 1.5
    label_type: str = "soft" # "soft" | "hard"
    augment_minority_only: bool = False  # True면 White(0) 제외 인종만 train 증강

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

        self.tf_train = build_transforms(train=True, size=cfg.img_size)
        self.tf_eval  = build_transforms(train=False, size=cfg.img_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            age, gender, race = _parse_utkface_filename(path)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.paths))

        img_np = _imread_rgb(path)   # HWC RGB ndarray
        img_pil = to_pil(img_np)     # PIL.Image

        # 증강 규칙 (train 시 비백인만 증강 옵션)
        if self.cfg.split == "train":
            if self.cfg.augment_minority_only:
                img = self.tf_train(img_pil) if race != 0 else self.tf_eval(img_pil)
            else:
                img = self.tf_train(img_pil)     # 기본은 훈련 전체 증강
        else:
            img = self.tf_eval(img_pil)

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
):
    all_paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
    if len(all_paths) == 0:
        all_paths = sorted(glob.glob(os.path.join(root, "*.png")))
    assert len(all_paths) > 100, f"[UTKFace] No images under: {root}"

    train_list, val_list = _stable_split(all_paths, val_ratio=val_ratio, seed=seed)

    train_ds = UTKFaceDataset(UTKFaceCfg(
        root=root,
        split="train",
        img_size=img_size,
        num_bins=num_bins,
        sigma=sigma,
        label_type=label_type,
        augment_minority_only=augment_minority_only
    ), file_list=train_list)

    val_ds = UTKFaceDataset(UTKFaceCfg(
        root=root,
        split="val",
        img_size=img_size, num_bins=num_bins,
        sigma=sigma,
        label_type=label_type,
        augment_minority_only=False
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