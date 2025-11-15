# faceage/data/datasets.py
import os, glob, hashlib
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from .transforms import build_transforms, to_pil
from faceage.data.constants import AGE_GROUPS, RACE_NAMES, NUM_RACES

def age_to_group_name(age: int) -> str:
    for name, lo, hi in AGE_GROUPS:
        if lo <= age <= hi:
            return name
    return "others"

def _parse_utkface_filename(path):
    base = os.path.basename(path)
    stem = base.split(".")[0]
    age, gender, race = map(int, stem.split("_")[:3])
    return age, gender, race

def _soft_label(age: int, num_bins: int = 91, sigma: float = 1.5) -> torch.Tensor:
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

# max_age(90)세 초과 샘플 제거
def _filter_by_age(paths: List[str], max_age: int = 90) -> List[str]:
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
    num_bins: int = 91
    sigma: float = 1.5
    label_type: str = "soft" # "soft" | "hard"
    augment_minority_only: bool = False  # True면 White(0) 제외 인종만 train 증강

    aug_strength: str = "medium"
    use_age_group_aug: bool = False
    aug_dup: int = 1

    use_random_erase: bool = False
    erase_prob: float = 0.2
    use_age_group_aug_dup: bool = False

class UTKFaceDataset(Dataset):
    def __init__(self, cfg: UTKFaceCfg, file_list: Optional[List[str]] = None):
        self.cfg = cfg

        # --- 파일 리스트 준비 ---
        if file_list is None:
            all_files = sorted(glob.glob(os.path.join(cfg.root, "*.jpg")))
            if len(all_files) == 0:
                all_files = sorted(glob.glob(os.path.join(cfg.root, "*.png")))
            assert len(all_files) > 100, f"[UTKFace] No images under: {cfg.root}"
            self.paths = all_files
        else:
            self.paths = file_list

        # --- 기본 transform 설정 ---
        self.tf_train = build_transforms(
            train=True,
            size=cfg.img_size,
            strength=cfg.aug_strength,
            use_random_erase=cfg.use_random_erase,
            erase_prob=cfg.erase_prob,
        )
        self.tf_eval = build_transforms(
            train=False,
            size=cfg.img_size,
        )

        # --- 나이대별 transform (강도 다르게) ---
        self.age_group_transforms = None
        if cfg.use_age_group_aug and cfg.split == "train":
            self.age_group_transforms = {}
            for name, lo, hi in AGE_GROUPS:
                if hi <= 19:          # 00–09, 10–19
                    strength = "medium"
                elif hi <= 39:        # 20–29, 30–39
                    strength = "weak"
                elif hi <= 59:        # 40–49, 50–59
                    strength = "medium"
                else:                 # 60–69, 70–79, 80–89
                    strength = "strong"

                self.age_group_transforms[name] = build_transforms(
                    train=True,
                    size=cfg.img_size,
                    strength=strength,
                    use_random_erase=cfg.use_random_erase,
                    erase_prob=cfg.erase_prob,
                )

        # --- 나이대별 dup 설정 (몇 배씩 뽑을지) ---
        # 기본 맵: tail(고령)일수록 많이 뽑도록 설계
        self.age_group_dup = {
            "00-09": 1,
            "10-19": 1,
            "20-29": 1,
            "30-39": 1,
            "40-49": 2,
            "50-59": 2,
            "60-69": 4,
            "70-79": 4,
            "80-89": 4,
        }

        # use_age_group_aug_dup=False면, 전 나이대 공통 dup만 사용
        # (cfg.aug_dup가 1이면 사실상 dup 없음)
        self.global_dup = max(1, int(cfg.aug_dup))

        # --- index_map 생성: 여기서 진짜 "나이대별로 많이 뽑기" 구현 ---
        self.index_map = []

        if self.cfg.split != "train":
            # val/test는 항상 1배, 순수한 평가용
            self.index_map = list(range(len(self.paths)))
        else:
            if cfg.use_age_group_aug_dup:
                # 🔥 나이대별 dup 적용 모드
                for i, path in enumerate(self.paths):
                    age, _, _ = _parse_utkface_filename(path)
                    group = age_to_group_name(age)
                    dup = self.age_group_dup.get(group, 1)  # 매핑 없으면 1배

                    # 필요하면 global_dup 도 곱할 수 있음 (지금은 age map만 사용)
                    # dup = dup * self.global_dup

                    for _ in range(dup):
                        self.index_map.append(i)
            else:
                # 🔥 전 나이대 공통 dup만 적용 (기존 aug_dup=K 의미 그대로)
                for i in range(len(self.paths)):
                    for _ in range(self.global_dup):
                        self.index_map.append(i)

        # sanity check
        assert len(self.index_map) > 0, "[UTKFaceDataset] index_map is empty!"

    def __len__(self):
        # 이제는 index_map 길이가 곧 dataset 길이
        return len(self.index_map)

    def __getitem__(self, idx):
        # index_map을 통해 실제 원본 인덱스로 매핑
        real_idx = self.index_map[idx]
        path = self.paths[real_idx]

        age, gender, race = _parse_utkface_filename(path)
        img_np = _imread_rgb(path)
        img_pil = to_pil(img_np)

        # 어떤 transform을 쓸지 결정
        if self.cfg.split == "train":
            if self.age_group_transforms is not None:
                group = age_to_group_name(age)
                tf = self.age_group_transforms.get(group, self.tf_train)
            else:
                tf = self.tf_train
        else:
            tf = self.tf_eval

        img = tf(img_pil)

        # label 생성
        if self.cfg.label_type == "soft":
            label = _soft_label(age, self.cfg.num_bins, self.cfg.sigma)
        else:
            label = torch.zeros(self.cfg.num_bins, dtype=torch.float32)
            label[min(int(age), self.cfg.num_bins - 1)] = 1.0

        race_oh = _race_one_hot(race)
        return img.float(), label.float(), race_oh, torch.tensor(age, dtype=torch.long)
    

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
    num_bins: int = 91,
    sigma: float = 1.5,
    label_type: str = "soft",
    augment_minority_only: bool = False,
    val_ratio: float = 0.2,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    max_age: int = 90,
    aug_strength: str = "medium",
    use_age_group_aug: bool = False,
    aug_dup: int = 1,
    use_random_erase: bool = False,
    erase_prob: float = 0.2,
    use_age_group_aug_dup: bool = False,

):
    all_paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
    if len(all_paths) == 0:
        all_paths = sorted(glob.glob(os.path.join(root, "*.png")))
    assert len(all_paths) > 100, f"[UTKFace] No images under: {root}"

    # 90세 초과 제거
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
        aug_strength=aug_strength,
        use_age_group_aug=use_age_group_aug,
        aug_dup=aug_dup,
        use_random_erase=use_random_erase,
        erase_prob=erase_prob,
        use_age_group_aug_dup=use_age_group_aug_dup,
    ), file_list=train_list)

    val_ds = UTKFaceDataset(UTKFaceCfg(
        root=root,
        split="val",
        img_size=img_size,
        num_bins=num_bins,
        sigma=sigma,
        label_type=label_type,
        augment_minority_only=False,
        aug_strength="none",
        use_age_group_aug=False,
        aug_dup=1,
        use_random_erase=False,
        erase_prob=0.0,
        use_age_group_aug_dup=False,
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