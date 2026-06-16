"""
faceage/train.py
얼굴 나이 추정 모델 학습 스크립트
- 백본(VGG / ResNet) + SoftHead(soft label regression) 구조
- Hard / Soft 라벨, CE / MSE / KLD 손실 선택 가능
- WandB 로깅, Early Stopping, 나이대별 MAE 계산 등 포함
"""

import os
import argparse
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm

from faceage.data.datasets import build_dataloaders
from faceage.models.model_factory import build_model
from faceage.models.head import SoftHead
from faceage.utils.seed import set_seed
from faceage.data.constants import AGE_GROUPS


"""
함수 이름: train_one_epoch
기능: epoch 동안 모델을 학습하고 평균 loss와 MAE를 반환
파라미터: 
1) model : nn.Module 백본 네트워크 (feature extractor)
2) head : nn.Module SoftHead (bin classification -> soft age regression)
3) loader : DataLoader 훈련 데이터 로더
4) criterion : nn.Module 손실 함수
5) optimizer : torch.optim.Optimizer 파라미터 업데이트에 사용할 옵티마이저
6) device : torch.device 'cuda' 또는 'cpu'
7) label_type : str "hard" (one-hot) 또는 "soft" (gaussian soft label)
8) loss_fn : str "ce" / "mse" / "kld"
9) num_bins : int 나이 bin 개수 (기본 91 → 0~90세)
10) use_race : bool, default=False race one-hot 벡터를 head 입력에 결합(concat) 할지 여부
리턴값 : Tuple[float, float]
        (epoch 평균 loss, epoch 평균 MAE)
"""
def train_one_epoch(model, head, loader, criterion, optimizer, device, label_type: str, loss_fn: str, num_bins: int, use_race: bool = False):
    model.train()
    total_loss, total_mae = 0.0, 0.0
    count = 0

    # 0 ~ num_bins-1 까지의 실수형 bin 중심값 (예: 0,1,2,...,90)
    bins = torch.arange(num_bins, device=device, dtype=torch.float32)

    for imgs, labels, races, ages in tqdm(loader, desc="Train", leave=False):
        imgs, labels, ages = imgs.to(device), labels.to(device), ages.to(device)
        races = races.to(device) if use_race else None   # race 사용 여부 결정

        optimizer.zero_grad(set_to_none=True) # 이전 batch의 gradient 초기화

        # Feature extraction
        feats = model(imgs) # (Batch, feat_dim)

        # Classification logits (bin 개수만큼)
        logits = head(feats, races) # (Batch, num_bins)

        # Loss 계산
        if label_type == "hard" and loss_fn == "ce": # CrossEntropyLoss
            targets = labels.argmax(dim=1)
            loss = criterion(logits, targets)
        elif label_type == "soft" and loss_fn == "kld": # KLD
            log_probs = torch.log_softmax(logits, dim=1) 
            loss = criterion(log_probs, labels)
        elif label_type == "soft" and loss_fn == "mse": # MSE
            probs = torch.softmax(logits, dim=1)
            loss = criterion(probs, labels)
        else:
            raise ValueError(f"Incompatible combo: label_type={label_type}, loss_fn={loss_fn}")

        # MAE 계산 (expected age)
        probs = F.softmax(logits, dim=1)
        pred_age = (probs * bins).sum(dim=1)    # E[age] = Σ p_i * i
        mae = (pred_age - ages.float()).abs().mean()

        # 역전파 & 파라미터 업데이트
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += mae.item()
        count += 1

    return total_loss / count, total_mae / count

"""
함수 이름: validate
기능 : 검증 데이터로 loss, 전체 MAE, 나이대별 MAE를 계산하여 반환
리턴값: Tuple[float, float, Dict[str, float]]
        (val_loss, val_mae, {age_group_name: mae})
"""
@torch.no_grad()
def validate(model, head, loader, criterion, device, label_type: str, loss_fn: str, num_bins: int, use_race: bool = False):
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    count = 0
    bins = torch.arange(num_bins, device=device, dtype=torch.float32)

    # 나이대별 누적 오차 및 샘플 수
    age_group_abs_err = {name: 0.0 for name, _, _ in AGE_GROUPS}
    age_group_count   = {name: 0   for name, _, _ in AGE_GROUPS}

    for imgs, labels, races, ages in tqdm(loader, desc="Val", leave=False):
        imgs, labels, ages = imgs.to(device), labels.to(device), ages.to(device)
        races = races.to(device) if use_race else None   # race 사용 여부 결정

        feats  = model(imgs)
        logits = head(feats, races)

        # Loss
        if label_type == "hard" and loss_fn == "ce":
            targets = labels.argmax(dim=1)
            loss = criterion(logits, targets)

        elif label_type == "soft" and loss_fn == "kld":
            loss = criterion(F.log_softmax(logits, dim=1), labels)

        elif label_type == "soft" and loss_fn == "mse":
            loss = criterion(F.softmax(logits, dim=1), labels)

        else:
            raise ValueError(f"Incompatible combo: label_type={label_type}, loss_fn={loss_fn}")

        # MAE (expected age)
        probs = F.softmax(logits, dim=1)
        pred_age = (probs * bins).sum(dim=1)
        mae = (pred_age - ages.float()).abs().mean()

        total_loss += loss.item()
        total_mae += mae.item()
        count += 1

        # 나이대별 MAE 누적
        per_sample_err = (pred_age - ages.float()).abs()
        for name, lo, hi in AGE_GROUPS:
            mask = (ages >= lo) & (ages <= hi)
            if mask.any():
                age_group_abs_err[name] += per_sample_err[mask].sum().item()
                age_group_count[name]   += mask.sum().item()

    # 나이대별 평균 MAE 계산 (루프 끝난 뒤에 한 번만 계산)
    mae_by_age_group = {}
    for name in age_group_abs_err:
        if age_group_count[name] > 0:
            mae_by_age_group[name] = age_group_abs_err[name] / age_group_count[name]

    return total_loss / count, total_mae / count, mae_by_age_group

"""
함수 이름: main
기능 : 전체 학습 파이프라인 실행
주요 동작 흐름:
    1. argparse로 훈련 설정 파싱 (데이터 경로, 에포크, 배치 사이즈, 학습률, 모델 종류 등)
    2. 실험 결과 저장 디렉토리 생성 및 랜덤 시드 고정
    3. WandB 실험 로깅 초기화 (run  이름 자동 생성)
    4. build_dataloaders()로 훈련, 검증 DataLoader 생성
    5. 백본 모델 (VGG or ResNet) + SoftHead 구성 및 GPU 이동
    6. 손실함수 (CE/MSE/KLD), 옵티마이저(Adam/AdamW/SGD 설정)
    7. 에포크 반복 :
        - train_one_epoch() 로 훈련
        - validate()로검증 및 나이대별 MAE 계산
        - Val loss 기준 Early Stopping 및 Best 모델 저장
        - WandB에 loss, MAE, learning rate, 나이대별 MAE 등 로깅
        - 5에포크 마다 체크포인트 저장
    8. 훈련 종료 후 WandB 세션 종료
"""
def main():
    # argparse 정의
    parser = argparse.ArgumentParser(description="Face-Age Training")
    # --- data ---
    parser.add_argument("--data_root", type=str, required=True) # data 경로
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/face-age/checkpoints") # checkpoint 저장 경로
    parser.add_argument("--augment_minority_only", action="store_true") # data 불균형 해결용 옵션
    parser.add_argument("--wandb_project", type=str, default="face-age") # wandb project name

    # --- Model(type, feat_dim, width, activation, droput, patience) ---
    parser.add_argument("--model_type", type=str, default="vgg", choices=["vgg", "resnet"])
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "leakyrelu", "gelu", "elu"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=8)

    # --- Optimizer ---
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # --- Loss ---
    parser.add_argument("--label_type", type=str, default="hard", choices=["hard", "soft"], help="라벨 생성 방식 (hard=one-hot, soft=gaussian soft label)")
    parser.add_argument("--loss_fn", type=str, default="ce", choices=["ce", "mse", "kld"],help="손실함수 선택 (ce=CrossEntropy, mse=MSE, kld=KLDiv)")
    parser.add_argument("--sigma", type=float, default=1.5, help="soft label 가우시안 폭 (sigma). label_type='soft'일 때만 사용됨.")

    # --- race_one_hot_vector ---
    parser.add_argument("--use_race_onehot", action="store_true", default=False)

    # Data Augmentation
    parser.add_argument("--aug_strength", type=str, default="medium", choices=["none", "weak", "medium", "strong"], help="훈련 데이터 공통 증강 강도")
    parser.add_argument("--use_age_group_aug", action="store_true", help="나이대별로 서로 다른 증강을 사용할지 여부")
    parser.add_argument("--aug_dup", type=int, default=1, help="훈련 데이터 한 이미지당 augmentation 샘플 개수 배수")

    parser.add_argument("--use_random_erase", action="store_true", help="RandomErasing을 사용할지 여부")
    parser.add_argument("--erase_prob", type=float, default=0.2, help="RandomErasing 적용 확률 (0.0 ~ 1.0)")
    parser.add_argument("--use_age_group_aug_dup", action="store_true", help="나이대별로 서로 다른 aug_dup을 적용할지 여부")
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    # Activation Function
    ACT = {
        "relu": nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(0.1),
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
    }[args.activation]

    # WandB run_name 자동 생성
    run_name = (
        f"{args.model_type.upper()}_"
        f"{args.loss_fn}_"
        f"act{args.activation}_"
        f"sig{args.sigma}_"
        f"drop{args.dropout}_"
        f"wd{args.weight_decay}_"
        f"aug{args.aug_strength}_"
        f"dup{args.aug_dup}"
    )

# 플래그: 켜져 있을 때만 붙임
    if args.use_age_group_aug:
        run_name += "_ageAug"

    if args.use_age_group_aug_dup:
        run_name += "_ageDup"

    if args.use_random_erase:
        run_name += f"_erase{args.erase_prob}"

    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    wandb.define_metric("epoch")
    wandb.define_metric("loss/*", step_metric="epoch")

    # --- Data loader ---
    try:
        train_loader, val_loader = build_dataloaders(
            root=args.data_root,
            batch_size=args.batch_size,
            img_size=200,
            num_bins=91,
            sigma=args.sigma,
            label_type=args.label_type,
            augment_minority_only=args.augment_minority_only,
            val_ratio=0.2,
            seed=42,
            max_age=90,
            aug_strength=args.aug_strength,
            use_age_group_aug=args.use_age_group_aug,
            aug_dup=args.aug_dup,
            use_random_erase=args.use_random_erase,
            erase_prob=args.erase_prob,
            use_age_group_aug_dup=args.use_age_group_aug_dup,
        )
        print(f"[DEBUG] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")
    except Exception as e:
        import traceback
        import glob
        print("[ERROR] build_dataloaders failed:", type(e).__name__, e)
        print("[DEBUG] data_root=", args.data_root)
        print("[DEBUG] some files:",
            glob.glob(os.path.join(args.data_root, "*"))[:5])
        traceback.print_exc()
        return

    # --- Model ---
    model = build_model(
        kind=args.model_type,
        in_channels=3,
        feat_dim=args.feat_dim,
        width=args.width,
        activation=ACT,
        dropout=args.dropout
    ).to(device)

    head = SoftHead(args.feat_dim, num_bins=91, use_race=args.use_race_onehot,).to(device)

    # --- Loss ---
    if args.loss_fn == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_fn == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss_fn == "kld":
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
    
    # --- Optimizer ---
    params = list(model.parameters()) + list(head.parameters())
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay, nesterov=True)

    # --- Training ---
    best_val = float("inf")  # 지금까지 최소 val_loss
    bad_epochs = 0           # 개선 없는 epoch 수

    # 총 Parameter 수
    params_total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    params_train = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
                sum(p.numel() for p in head.parameters() if p.requires_grad)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = train_one_epoch(model, head, train_loader, criterion, optimizer, device, label_type=args.label_type, loss_fn=args.loss_fn, num_bins=91, use_race=args.use_race_onehot)
        val_loss, val_mae, val_mae_by_age = validate(model, head, val_loader, criterion, device, label_type=args.label_type, loss_fn=args.loss_fn, num_bins=91, use_race=args.use_race_onehot)

        current_lr = optimizer.param_groups[0]["lr"]

        # 콘솔 출력
        print(f"[{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_mae={train_mae:.2f}, val_mae={val_mae:.2f}, lr={current_lr:.2e}")

        # Early Stopping (Val Loss 기준)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            bad_epochs = 0

            # best checkpoint 저장
            best_ckpt = os.path.join(args.save_dir, f"best_{args.model_type}_W{args.width}_F{args.feat_dim}.pt")
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, best_ckpt)
            print(f"New best model saved (val_loss={val_loss:.4f}) → {best_ckpt}")
            wandb.log({"checkpoint/best_epoch": epoch,
                   "checkpoint/best_val_loss": val_loss,
                   "checkpoint/best_val_mae": val_mae})
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break    
        
        mae_by_age_logs = {f"mae_by_age/{k}": v for k, v in val_mae_by_age.items()}

        # W&B 로깅 
        wandb.log({
            "epoch": epoch,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "mae/train": train_mae,
            "mae/val": val_mae,
            "lr": current_lr,
            "params/total": params_total,
            "params/trainable": params_train,
            **mae_by_age_logs,
        })

        # epoch 5번마다 Checkpoint 저장
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(
                args.save_dir, f"{args.model_type}_W{args.width}_F{args.feat_dim}_E{epoch}.pt"
            )
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, ckpt_path)
            print(f"Saved checkpoint → {ckpt_path}")
            wandb.log({"checkpoint/saved_epoch": epoch})

    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()