import os
import argparse
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm
from datetime import datetime

from faceage.data.datasets import build_dataloaders
from faceage.models.cnn import SimpleCNN
from faceage.models.model_factory import build_model
from faceage.models.head import SoftHead
from faceage.utils.seed import set_seed

def train_one_epoch(model, head, loader, criterion, optimizer, device, label_type: str, loss_fn: str, num_bins: int, use_race: bool = False):
    model.train()
    total_loss, total_mae = 0.0, 0.0
    count = 0
    bins = torch.arange(num_bins, device=device, dtype=torch.float32)

    for imgs, labels, races, ages in tqdm(loader, desc="Train", leave=False):
        imgs, labels, ages = imgs.to(device), labels.to(device), ages.to(device)
        races = races.to(device) if use_race else None   # race 사용 여부 결정

        optimizer.zero_grad(set_to_none=True) # 이전 batch의 gradient 초기화

        feats = model(imgs) # (Batch, feat_dim)
        logits = head(feats, races) # (Batch, num_bins)

        # Loss
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

        # MAE
        probs = F.softmax(logits, dim=1)
        pred_age = (probs * bins).sum(dim=1)
        mae = (pred_age - ages.float()).abs().mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mae += mae.item()
        count += 1

    return total_loss / count, total_mae / count


@torch.no_grad()
def validate(model, head, loader, criterion, device, label_type: str, loss_fn: str, num_bins: int, use_race: bool = False):
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    count = 0
    bins = torch.arange(num_bins, device=device, dtype=torch.float32)

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

        # === MAE ===
        probs = F.softmax(logits, dim=1)
        pred_age = (probs * bins).sum(dim=1)
        mae = (pred_age - ages.float()).abs().mean()

        total_loss += loss.item()
        total_mae += mae.item()
        count += 1

    return total_loss / count, total_mae / count


def main():
    parser = argparse.ArgumentParser(description="Face-Age Training")
    parser.add_argument("--data_root", type=str, required=True) # data 경로
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/face-age/checkpoints") # checkpoint 저장 경로
    parser.add_argument("--augment_minority_only", action="store_true") # data 불균형 해결용 옵션
    parser.add_argument("--wandb_project", type=str, default="face-age") # wandb project name

    # --- Model(type, feat_dim, width, activation, droput, patience) ---
    parser.add_argument("--model_type", type=str, default="vgg", choices=["vgg", "resnet"])
    parser.add_argument("--feat_dim", type=int, default=128)   # ✅ 추가
    parser.add_argument("--width", type=int, default=64)       # ✅ 추가
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

    # WandB run_name
    run_name = (
        f"{args.model_type.upper()}_"
        f"{args.loss_fn}_"
        f"{args.activation}_"
        f"sig{args.sigma}_"
        f"drop{args.dropout}_"
        f"wd{args.weight_decay}"
    )
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    wandb.define_metric("epoch")
    wandb.define_metric("loss/*", step_metric="epoch")

    # --- Data ---
    try:
        train_loader, val_loader = build_dataloaders(
            root=args.data_root,                # "/content/UTKFace/UTKFace"
            batch_size=args.batch_size,
            img_size=200,
            num_bins=86,
            sigma=args.sigma,
            label_type=args.label_type,
            augment_minority_only=args.augment_minority_only,
            val_ratio=0.2,
            seed=42,
            max_age=85,
        )
        print(f"[DEBUG] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")
    except Exception as e:
        import traceback, glob
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

    head = SoftHead(args.feat_dim, num_bins=86, use_race=args.use_race_onehot,).to(device)

    # Loss
    if args.loss_fn == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_fn == "mse":
        criterion = torch.nn.MSELoss()
    elif args.loss_fn == "kld":
        criterion = torch.nn.KLDivLoss(reduction="batchmean")
    
    # Optimizer
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
        train_loss, train_mae = train_one_epoch(model, head, train_loader, criterion, optimizer, device, label_type=args.label_type, loss_fn=args.loss_fn, num_bins=86, use_race=args.use_race_onehot)
        val_loss, val_mae = validate(model, head, val_loader, criterion, device, label_type=args.label_type, loss_fn=args.loss_fn, num_bins=86, use_race=args.use_race_onehot)

        current_lr = optimizer.param_groups[0]["lr"]

        # 콘솔 출력
        print(f"[{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_mae={train_mae:.2f}, val_mae={val_mae:.2f}, lr={current_lr:.2e}")

        # === Early Stopping (Val Loss 기준) ===
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
            print(f"🌟 New best model saved (val_loss={val_loss:.4f}) → {best_ckpt}")
            wandb.log({"checkpoint/best_epoch": epoch,
                   "checkpoint/best_val_loss": val_loss,
                   "checkpoint/best_val_mae": val_mae})
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"⏹ Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break    
        
        # === W&B 로깅 ===
        wandb.log({
            "epoch": epoch,
            "loss/train": train_loss,
            "loss/val": val_loss,
            "mae/train": train_mae,
            "mae/val": val_mae,
            "train/val_loss_gap": abs(train_loss - val_loss),
            "lr": current_lr,
            "params/total": params_total,
            "params/trainable": params_train,
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
            print(f"💾 Saved checkpoint → {ckpt_path}")
            wandb.log({"checkpoint/saved_epoch": epoch})

    wandb.finish()
    print("✅ Training complete!")

if __name__ == "__main__":
    main()