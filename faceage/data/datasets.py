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
    """
    함수 이름: age_to_group_name
    기능: age(int)를 받아서 AGE_GROUPS에 정의된 나이대 이름(str)으로 바꾼다. (ex: 23 > 20-29)
    파라미터:
        age (int): 실제 나이
    리턴값:
        name (str): 해당 나이가 속하는 나이대 이름, 없으면 "others" 리턴.
    """
    for name, lo, hi in AGE_GROUPS:
        if lo <= age <= hi:
            return name
    return "others"


def _parse_utkface_filename(path):
    """
    함수 이름: _parse_utkface_filename
    기능: UTKFace 이미지 파일 경로에서 파일명을 파싱하여 (age, gender, race)를 뽑는다.
        (파일명 형식: age_gender_race_....jpg)
    파라미터:
        path (str): 이미지 파일 경로
    리턴값:
        (int, int, int): (age, gender, race)
    """
    base = os.path.basename(path)      # 경로에서 이미지 파일명만 떼옴
    stem = base.split(".")[0]          # 확장자(.jpg) 제거
    age, gender, race = map(int, stem.split("_")[:3])  # 파일명에서 앞 세 개를 age/gender/race로 사용
    return age, gender, race


def _soft_label(age: int, num_bins: int = 91, sigma: float = 1.5) -> torch.Tensor:
    """
    함수 이름: _soft_label
    기능: 나이(정수값)를 가우시안 형태의 soft label 분포로 바꿔준다.
        (예: 값이 23이면 23 근처 bin에도 확률 조금씩 줌)
    파라미터:
        age (int): 실제 나이
        num_bins (int): 나이 bin 개수 (기본 91 → 0~90)
        sigma (float): 가우시안 폭 (클수록 더 완만하게 퍼짐)
    리턴값:
        torch.Tensor: (num_bins,) 크기의 1차원 텐서, 확률 분포 (합 ≈ 1)
    """
    # 나이가 범위 밖으로 나가면 0 ~ num_bins-1 사이로 클리핑(예: 값이 95 > 90으로 클리핑)
    age = max(0, min(num_bins - 1, int(age)))
    xs = np.arange(num_bins, dtype=np.float32)         # [0, 1, ..., num_bins-1]  정수 배열 
    g = np.exp(-0.5 * ((xs - age) / sigma) ** 2)       # xs 배열에서 각 값 i에 대해 가우시안 계산
    g /= g.sum() + 1e-8                                # 합이 1이 되도록 정규화
    return torch.from_numpy(g)                         # numpy 배열을 torch 텐서로 변환해서 반환.


def _race_one_hot(race: int) -> torch.Tensor:
    """
    함수 이름: _race_one_hot
    기능: 인종 인덱스를 one-hot 벡터로 바꿔준다.
        (예: 인종 인덱스가 2 > (0, 0, 1, 0, 0)의 one-hot 텐서 반환)
    파라미터:
        race (int): 인종 인덱스
    리턴값:
        torch.Tensor: (NUM_RACES,) 크기의 one-hot 텐서
    """
    idx = min(max(int(race), 0), NUM_RACES - 1)         # 인종 인덱스가 범위를 벗어나면 클리핑
    v = torch.zeros(NUM_RACES, dtype=torch.float32)     # 크기가 NUM_RACES인 텐서 생성
    v[idx] = 1.0                                        # 해당 인종 인덱스 위치에만 값 1.0 설정
    return v


def _imread_rgb(path: str) -> np.ndarray:
    """
    함수 이름: _imread_rgb
    기능: 이미지 파일을 읽어서 RGB numpy 배열(H, W, C)로 반환한다.
    파라미터:
        path (str): 이미지 경로
    리턴값:
        np.ndarray: RGB 이미지 (HWC)
    """
    # PIL로 읽어서 RGB로 통일
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _stable_split(paths: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    함수 이름: _stable_split
    기능: 파일 경로 리스트를 train/val로 나누는데,
         랜덤이 아니라 파일 경로 + seed를 md5 해시로 정렬해서
         항상 같은 방식으로 나누도록 만든다. > 항상 똑같은 train/val 결과 나옴.
    파라미터:
        paths (List[str]): 전체 이미지 경로 리스트
        val_ratio (float): validation 비율 
        seed (int): 해시에 섞을 시드 값
    리턴값:
        (List[str], List[str]): (train_paths, val_paths)를 튜플로 반환
    """
    keys = []
    for p in paths:
        # path + seed 문자열에 대해 md5 해시 계산
        h = hashlib.md5((p + str(seed)).encode()).hexdigest()
        keys.append(int(h, 16))  # 16진수 해시를 10진수 int로 변환
    # 해시 값 기준으로 오름차순 정렬된 인덱스
    order = np.argsort(keys)
    n = len(paths)                       # 전체 파일 개수
    n_val = int(round(n * val_ratio))    # val로 보낼 개수
    val_idx = set(order[:n_val])         # 정렬된 인덱스 중 앞부분(n_val개)을 val 셋으로 사용
    train, val = [], []
    for i, p in enumerate(paths):
        # 인덱스가 val_idx에 있으면 val, 아니면 train
        (val if i in val_idx else train).append(p)
    return train, val


# max_age(90)세 초과 샘플 제거
def _filter_by_age(paths: List[str], max_age: int = 90) -> List[str]:
    """
    함수 이름: _filter_by_age
    기능: 파일 리스트에서 age가 max_age보다 큰 샘플은 제거한다.
    파라미터:
        paths (List[str]): 전체 이미지 경로 리스트
        max_age (int): 허용하는 최대 나이 (기본 90)
    리턴값:
        List[str]: 필터링 후 남은 경로 리스트
    """
    kept = []
    dropped = 0
    for p in paths:
        try:
            age, _, _ = _parse_utkface_filename(p)  # 파일명에서 age 뽑음
            if age <= max_age:
                kept.append(p)
            else:
                dropped += 1 
        except Exception:
            # 파일명 파싱 실패한 건 그냥 스킵
            continue

    print(f"[UTKFace] Removed {dropped} images with age > {max_age}")
    # 필터링 후 샘플이 너무 적으면 assert로 막아둠
    assert len(kept) > 100, f"[UTKFace] Too few samples after filtering (<= {max_age}yrs)"
    return kept


@dataclass
class UTKFaceCfg:
    """
    클래스 이름: UTKFaceCfg
    기능: UTKFaceDataset에서 쓸 설정들을 한 군데에 모아 둔 설정용 클래스
    """
    root: str                           # 데이터 루트 경로
    split: str = "train"                # "train" | "val"
    img_size: int = 200                 # 리사이즈 할 이미지 크기
    num_bins: int = 91                  # age bin 개수
    sigma: float = 1.5                  # soft label 가우시안 폭
    label_type: str = "soft"            # "soft" | "hard"
    augment_minority_only: bool = False # True면 White(0) 제외 인종만 train 증강

    aug_strength: str = "medium"        # 기본 증강 강도, "weak" | "medium" | "strong" | "none"
    use_age_group_aug: bool = False     # 나이대 별로 다른 transform 쓸지 여부
    aug_dup: int = 1                    # 전체 데이터 중복 배수

    use_random_erase: bool = False      # Random erase 사용할지 여부
    erase_prob: float = 0.2             # Random erase 확률
    use_age_group_aug_dup: bool = False # 나이대 별 dup 설정을 쓸지 여부


class UTKFaceDataset(Dataset):
    """
    클래스 이름: UTKFaceDataset
    기능: UTKFace 이미지들을 PyTorch Dataset 형태로 감싸는 클래스.
         DataLoader에 넣어서 train/val으로 편하게 쓸 수 있게 만든다.
         - __len__: 전체 샘플 수 반환 (dup 포함)
         - __getitem__: 주어진 인덱스로 (이미지 텐서, 라벨, 인종 one-hot, 나이)의 샘플 하나를 뽑아줌
    """
    def __init__(self, cfg: UTKFaceCfg, file_list: Optional[List[str]] = None):
        """
        함수 이름: __init__
        기능: 설정(cfg)과 파일 리스트를 받아서 Dataset을 초기화한다.
             파일 경로, transform, age group별 중복(index_map) 등을 준비함.
        파라미터:
            cfg (UTKFaceCfg): 데이터셋 설정
            file_list (Optional[List[str]]): 미리 나눠진 파일 리스트 (train 혹은 val)
                                            None이면 root 아래에서 직접 찾음
        """
        self.cfg = cfg

        # 파일 리스트 준비
        if file_list is None:
            # root 아래의 jpg 파일들을 찾는다.
            all_files = sorted(glob.glob(os.path.join(cfg.root, "*.jpg")))
            # jpg가 하나도 없으면 png도 시도
            if len(all_files) == 0:
                all_files = sorted(glob.glob(os.path.join(cfg.root, "*.png")))
            # 데이터가 너무 적으면 에러
            assert len(all_files) > 100, f"[UTKFace] No images under: {cfg.root}"
            self.paths = all_files
        else:
            # 이미 분할된 리스트가 들어온 경우 그대로 사용
            self.paths = file_list

        # 기본 transform 설정
        # - train용 transform (증강 포함)
        self.tf_train = build_transforms(
            train=True,
            size=cfg.img_size,
            # 증강 옵션
            strength=cfg.aug_strength,
            use_random_erase=cfg.use_random_erase,
            erase_prob=cfg.erase_prob,
        )
        # - eval용 transform (증강 거의 없음)
        self.tf_eval = build_transforms(
            train=False,
            size=cfg.img_size,
        )

        # 나이대별 transform (강도 다르게) 
        self.age_group_transforms = None
        if cfg.use_age_group_aug and cfg.split == "train":
            # 나이대별로 다른 강도를 쓰기 위한 dict
            self.age_group_transforms = {}
            for name, lo, hi in AGE_GROUPS:
                # 나이대 구간에 따라 강도 다르게
                if hi <= 19:          # 00–09, 10–19
                    strength = "medium"
                elif hi <= 39:        # 20–29, 30–39
                    strength = "weak"
                elif hi <= 59:        # 40–49, 50–59
                    strength = "medium"
                else:                 # 60–69, 70–79, 80–89
                    strength = "strong"

                # 해당 나이대 이름(name)으로 transform 기록
                self.age_group_transforms[name] = build_transforms(
                    train=True,
                    size=cfg.img_size,
                    strength=strength,
                    use_random_erase=cfg.use_random_erase,
                    erase_prob=cfg.erase_prob,
                )

        # 나이대별 dup 설정
        # 나이가 많을수록 데이터가 적을 수 있으므로 tail 구간일수록 더 많이 복제하도록 설정.
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

        # use_age_group_aug_dup=False이면 전체 글로벌 배수만 적용
        # (aug_dup=1이면 사실상 복제 X)
        self.global_dup = max(1, int(cfg.aug_dup))

        # index_map 생성
        # index_map에 원본 인덱스를 중복해서 넣음으로써 샘플을 여러 번 나오게 하는 구조
        self.index_map = []

        if self.cfg.split != "train":
            # val/test는 증강용 복제 없이 1배
            self.index_map = list(range(len(self.paths)))
        else:
            if cfg.use_age_group_aug_dup:
                # 나이대별 dup 적용 옵션
                for i, path in enumerate(self.paths):
                    age, _, _ = _parse_utkface_filename(path) # 파일명에서 age 파싱
                    group = age_to_group_name(age)            # age > 나이대
                    dup = self.age_group_dup.get(group, 1)    # 매핑 없으면 1배

                    # 필요하면 dup에 global_dup를 곱해서 더 키울 수도 있음
                    # dup = dup * self.global_dup

                    # 해당 인덱스를 dup만큼 index_map에 추가
                    for _ in range(dup):
                        self.index_map.append(i)
            else:
                # 전 나이대 공통 dup만 적용 (aug_dup 값만큼 전체 데이터 늘림)
                for i in range(len(self.paths)):
                    for _ in range(self.global_dup):
                        self.index_map.append(i)

        # sanity check
        assert len(self.index_map) > 0, "[UTKFaceDataset] index_map is empty!"

    def __len__(self):
        """
        함수 이름: __len__
        기능: Dataset의 전체 길이를 반환한다.
             (실제 파일 개수가 아니라, dup까지 적용된 index_map 기준)
        파라미터:
            없음 (self만 사용)
        리턴값:
            int: 전체 샘플 개수
        """
        return len(self.index_map)

    def __getitem__(self, idx):
        """
        함수 이름: __getitem__
        기능: 주어진 인덱스에 해당하는 하나의 샘플을 반환한다.
             (이미지 텐서, age 라벨(soft/hard), race one-hot, 실제 age)
             DataLoader가 배치 만들 때, 내부적으로 계속 호출함.
        파라미터:
            idx (int): 0 ~ len(self)-1 범위의 인덱스
        리턴값:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                img.float(): (C, H, W) 이미지 텐서
                label.float(): (num_bins,) age label 벡터, label type (soft or hard)
                race_oh: (NUM_RACES,) 인종 one-hot 텐서
                age_tensor: scalar long 텐서 (실제 나이)
        """
        # index_map을 통해 실제 원본 인덱스로 매핑
        real_idx = self.index_map[idx]
        path = self.paths[real_idx]

        # 파일명에서 age, gender, race 추출
        age, gender, race = _parse_utkface_filename(path)
        # 이미지 읽기 (numpy HWC)
        img_np = _imread_rgb(path)
        # PIL 이미지로 변환
        img_pil = to_pil(img_np)

        # 어떤 transform을 쓸지 결정(train/val에 따라)
        if self.cfg.split == "train":
            # train일 때, 나이대별 transform이 있으면 그거 사용
            if self.age_group_transforms is not None:
                group = age_to_group_name(age)
                tf = self.age_group_transforms.get(group, self.tf_train)
            # 나이대별 transform이 없으면 공통 transform 사용
            else:
                tf = self.tf_train
        else:
            # val/test는 항상 eval transform 사용
            tf = self.tf_eval

        # 선택된 transform을 실제 이미지에 적용해서 최종 텐서 얻기
        img = tf(img_pil)

        # age label 생성 (soft / hard)
        if self.cfg.label_type == "soft":
            # soft label: 가우시안 분포
            label = _soft_label(age, self.cfg.num_bins, self.cfg.sigma)
        else:
            # hard label: one-hot
            label = torch.zeros(self.cfg.num_bins, dtype=torch.float32)
            label[min(int(age), self.cfg.num_bins - 1)] = 1.0

        # 인종 인덱스를 one-hot 벡터로 변환
        race_oh = _race_one_hot(race)

        # 이미지, 나이 라벨, 인종 one-hot, 실제 나이 텐서를 튜플로 반환
        return img.float(), label.float(), race_oh, torch.tensor(age, dtype=torch.long)
    

# ---------- DataLoader Builder ----------
def _seed_worker(worker_id):
    """
    함수 이름: _seed_worker
    기능: DataLoader에서 여러 worker를 쓸 때,
        각 worker마다 numpy와 random 모듈의 seed를 설정한다.
    파라미터:
        worker_id (int): worker 인덱스 (사실 여기서는 안 씀)
    리턴값:
        없음
    """
    import random
    # torch.initial_seed():base seed 값
    # 2**32로 나눠서 numpy/random seed로 사용
    np.random.seed(torch.initial_seed() % 2**32)
    random.seed(torch.initial_seed() % 2**32)


def _env_defaults():
    """
    함수 이름: _env_defaults
    기능: 지금 환경이 GPU(CUDA)인지 CPU/MPS인지 보고
         DataLoader 기본 옵션(num_workers, pin_memory)을 적당히 정해준다.
    파라미터:
        없음
    리턴값:
        dict: {"num_workers": int, "pin_memory": bool}
            "num_workers": DataLoader에서 사용할 worker 개수
            "pin_memory": pin_memory 옵션을 켤지 여부
    """
    # CUDA GPU 면 workers/pin_memory 상승, 그 외엔 MPS/CPU면 안전값
    if torch.cuda.is_available():
        return dict(num_workers=2, pin_memory=True)
    else:
        return dict(num_workers=0, pin_memory=False)


def build_dataloaders(
    root: str,                          # 이미지가 들어 있는 루트 디렉토리
    batch_size: int = 64,               # 배치 크기
    num_workers: Optional[int] = None,  # DataLoader worker 수 (None이면 환경 기본값 사용)
    img_size: int = 200,                # 리사이즈할 이미지 크기
    num_bins: int = 91,                 # age bin 개수
    sigma: float = 1.5,                 # soft label 가우시안 폭
    label_type: str = "soft",           # "soft" | "hard"
    augment_minority_only: bool = False,# 인종 불균형 관련 옵션 (현재 내부에서 직접 사용 X)
    val_ratio: float = 0.2,             # 전체 중 val 비율
    seed: int = 42,                     # split 및 DataLoader generator seed
    pin_memory: Optional[bool] = None,  # DataLoader pin_memory 설정 (None이면 환경 기본값)
    max_age: int = 90,                  # 이 나이보다 큰 샘플은 제거
    aug_strength: str = "medium",       # 기본 증강 강도
    use_age_group_aug: bool = False,    # 나이대별로 다른 transform 쓸지 여부
    aug_dup: int = 1,                   # 전체 데이터 dup 배수
    use_random_erase: bool = False,     # Random Erase 사용 여부
    erase_prob: float = 0.2,            # Random Erase 확률
    use_age_group_aug_dup: bool = False,# 나이대별 dup 설정 사용할지 여부
):
    """
    함수 이름: build_dataloaders
    기능: UTKFace 데이터셋을 위한 train/val DataLoader를 한 번에 만들어준다.
         - 파일 경로 모으기
         - 나이 기준 필터링
         - train/val split
         - UTKFaceDataset 생성
         - DataLoader 감싸서 반환
    리턴값:
        (DataLoader, DataLoader): (train_loader, val_loader)
    """
    # root 아래의 jpg 파일들을 모두 수집
    all_paths = sorted(glob.glob(os.path.join(root, "*.jpg")))
    # jpg가 없다면 png도 시도
    if len(all_paths) == 0:
        all_paths = sorted(glob.glob(os.path.join(root, "*.png")))
    # 데이터가 너무 적으면 에러
    assert len(all_paths) > 100, f"[UTKFace] No images under: {root}"

    # 90세 초과 제거 (max_age 기준 필터링)
    all_paths = _filter_by_age(all_paths, max_age=max_age)

    # 해시 기반으로 train/val 나누기 (항상 같은 split 되도록)
    train_list, val_list = _stable_split(all_paths, val_ratio=val_ratio, seed=seed)

    # Train Dataset 생성
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

    # Val Dataset 생성
    # 검증용은 증강/dup 거의 안 쓰고 순수 평가 용도로만 사용한다
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

    # num_workers나 pin_memory가 지정 안 되었다면 환경 기본값 사용
    if num_workers is None or pin_memory is None:
        d = _env_defaults()
        if num_workers is None:
            num_workers = d["num_workers"]
        if pin_memory is None:
            pin_memory = d["pin_memory"]

    # DataLoader 안에서 사용할 random generator (재현성 위해 seed 고정)
    g = torch.Generator()
    g.manual_seed(seed)

    # Train DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,              # train은 섞어서 사용
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=False
    )

    # Val DataLoader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,             # val은 순서 유지
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=False
    )

    # 두 개 다 반환
    return train_loader, val_loader
