import os
import argparse
import torch
import wandb
from tqdm import tqdm
from datetime import datetime

from faceage.data.datasets import build_dataloaders
from faceage.models.cnn import SimpleCNN
from faceage.models.head import SoftHead
from faceage.utils.seed import set_seed


def train_one_epoch(model, head, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, labels, races, ages in tqdm(loader, desc="Train", leave=False):
        imgs, labels, races = imgs.to(device), labels.to(device), races.to(device)
        optimizer.zero_grad()
        feats = model(imgs)
        logits = head(feats, races)
        loss = criterion(logits, labels.argmax(dim=1))
        loss.backward()
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
    parser.add_argument(
        "--save_dir", type=str, default="/content/drive/MyDrive/face-age/checkpoints"
    )
    parser.add_argument("--augment_minority_only", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="face-age")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # --- W&B init ---
    run = wandb.init(
        project=args.wandb_project, config=vars(args), job_type="train"
    )  # job_type에 train 명시

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(f"🔹 Run name: {run_name}")

    # --- Data ---
    train_loader, val_loader = build_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        augment_minority_only=args.augment_minority_only,
    )

    # --- Model ---
    model = SimpleCNN().to(device)
    head = SoftHead(128, num_bins=86).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=args.lr
    )

    # --- Training ---
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, head, train_loader, criterion, optimizer, device
        )
        val_loss = validate(model, head, val_loader, criterion, device)

        print(
            f"[{epoch}/{args.epochs}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"model_epoch{epoch}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"💾 Saved checkpoint → {ckpt_path}")

    print("✅ Training complete!")

    # --- Create Model Artifacts ---

    model_artifact = wandb.Artifact(
        name="face-age", type="model", description="Trained model weights"
    )
    model_artifact.add_file(ckpt_path)
    run.log_artifact(model_artifact)

    # --- Create Dataset Artifacts ---

    data_artifact = wandb.Artifact(
        name="raw_image_data",
        type="dataset",
        description="Initial dataset from source, before filtering",
    )
    data_artifact.add_dir("/Users/iseunghun/Desktop/ML/raw_image_data-v0")
    run.log_artifact(data_artifact)

    run.finish


if __name__ == "__main__":
    main()
