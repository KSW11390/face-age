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
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/face-age/checkpoints")
    parser.add_argument("--augment_minority_only", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="face-age")

    # === 추가 인자 ===
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "leakyrelu", "gelu", "elu"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--model_type", type=str, default="vgg", choices=["vgg", "resnet"])

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # --- W&B init ---
    run_name = f"{args.model_type.upper()}_W{args.width}_D{args.depth}_{args.activation}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    ckpt_path = os.path.join(
        args.save_dir,
        f"{args.model_type}_W{args.width}_D{args.depth}_E{epoch}.pt"
    )

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

    head = SoftHead(128, num_bins=86).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # --- Training ---
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, head, train_loader, criterion, optimizer, device)
        val_loss = validate(model, head, val_loader, criterion, device)

        print(f"[{epoch}/{args.epochs}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"model_epoch{epoch}.pt")
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, ckpt_path)
            print(f"💾 Saved checkpoint → {ckpt_path}")

    wandb.finish()
    print("✅ Training complete!")


if __name__ == "__main__":
    main()