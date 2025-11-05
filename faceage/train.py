import os
import argparse
import torch
import wandb
from tqdm import tqdm
from datetime import datetime

from faceage.data.datasets import build_dataloaders
from faceage.models.cnn import SimpleCNN
from faceage.models.model_factory import build_model
from faceage.models.head import SoftHead
from faceage.utils.seed import set_seed


def train_one_epoch(model, head, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels, races, ages in tqdm(loader, desc="Train", leave=False):
        imgs, labels, races = imgs.to(device), labels.to(device), races.to(device)
        optimizer.zero_grad() # 이전 batch의 gradient 초기화
        feats = model(imgs) # forwardpass
        logits = head(feats, races) # 모델 통해 추출한 feature에 인종 정보 결합
        loss = criterion(logits, labels.argmax(dim=1))
        loss.backward() # backprop
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, head, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for imgs, labels, races, ages in tqdm(loader, desc="Val", leave=False):
        imgs, labels, races = imgs.to(device), labels.to(device), races.to(device)
        feats = model(imgs)
        logits = head(feats, races)
        loss = criterion(logits, labels.argmax(dim=1))
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Face-Age Training")
    parser.add_argument("--data_root", type=str, required=True) # data 경로
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/face-age/checkpoints") # checkpoint 저장 경로
    parser.add_argument("--augment_minority_only", action="store_true") # data 불균형 해결용 옵션
    parser.add_argument("--wandb_project", type=str, default="face-age") # wandb project name

    # --- 모델/옵션 ---
    parser.add_argument("--model_type", type=str, default="vgg", choices=["vgg", "resnet"])
    parser.add_argument("--feat_dim", type=int, default=128)   # ✅ 추가
    parser.add_argument("--width", type=int, default=64)       # ✅ 추가
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "leakyrelu", "gelu", "elu"])
    parser.add_argument("--dropout", type=float, default=0.0)
    # --- 옵티마이저 ---
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0.0)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

   # ✅ ACT 정의
    ACT = {
        "relu": nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(0.1),
        "gelu": nn.GELU(),
        "elu": nn.ELU(),
    }[args.activation]

    # ✅ W&B run name (depth 제거, feat_dim/width 사용)
    run_name = f"{args.model_type.upper()}_W{args.width}_F{args.feat_dim}_{args.activation}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    # --- Data ---
    train_loader, val_loader = build_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        augment_minority_only=args.augment_minority_only,
    )

    # --- Model ---
    model = build_model(
        kind=args.model_type,
        in_channels=3,
        feat_dim=args.feat_dim,
        width=args.width,
        depth=args.depth,
        activation=ACT,
    ).to(device)

    head = SoftHead(args.feat_dim, num_bins=86).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    params = list(model.parameters()) + list(head.parameters())
    
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:  # sgd
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay, nesterov=True)
    # --- Training ---
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, head, train_loader, criterion, optimizer, device)
        val_loss = validate(model, head, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        # 콘솔 출력
        print(f"[{epoch}/{args.epochs}] "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={current_lr:.2e}")

        # === W&B 로깅 ===
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/val_loss_gap": abs(train_loss - val_loss),
            "lr": current_lr,
            "params/total": sum(p.numel() for p in model.parameters()) +
                            sum(p.numel() for p in head.parameters()),
            "params/trainable": sum(p.numel() for p in model.parameters() if p.requires_grad) +
                                sum(p.numel() for p in head.parameters() if p.requires_grad)
        })

        # ✅ ckpt_path는 루프 안에서 epoch로 생성
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"{args.model_type}_W{args.width}_F{args.feat_dim}_E{epoch}.pt")
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