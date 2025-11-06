import os
import argparse
import torch
import wandb
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
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

@hydra.main(version_base=None, config_path="config", config_name="train/default")
def main(cfg: DictConfig):
    parser = argparse.ArgumentParser(description="Face-Age Training")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--save_dir", type=str, default="/content/drive/MyDrive/MLProject/checkpoints"
    )
    parser.add_argument("--augment_minority_only", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="face-age")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")


    name_parts = [
        f"LR{args.lr}",
        f"BS{args.batch_size}"
    ]
    if args.augment_minority_only:
        name_parts.append("AugOnly")

    run_name = "_".join(name_parts)

    # --- W&B init ---
    run = wandb.init(
        project="face-age",
        name=run_name,
        config=vars(args),
        job_type="train"
    ) 

    # --- Model ---
    model = SimpleCNN(in_channels=cfg.model.in_channels).to(device)
    head = SoftHead(
        in_dim=128, 
        num_bins=86, 
        num_bins=cfg.data.num_bins,
        use_race=cfg.data.use_race_onehot,
        dropout=cfg.model.dropout,).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()), lr=args.lr
    )

    # --- Model 아티팩트 사용 시도 ---
    model_path = "HongikML/face-age/face-age-checkpoints:latest" 
    start_epoch = 1
    try:
        model_dir = run.use_artifact(model_path, type="model").download()
        model_path = f"{model_dir}/best_model.pt"

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        head.load_state_dict(checkpoint['head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

        print(f"✅ Checkpoint loaded. Resuming from epoch {start_epoch}")
    except Exception as e:
        print("❌ Artifact가 존재하지 않음. 새로운 모델 훈련 시작.")
        start_epoch = 1

    # --- Dataset 아티팩트 사용 시도 ---
    datatset_path = "HongikML/face-age/image_data:latest"
    dataset_artifact_exists = True
    try:
        dataset_dir = run.use_artifact(datatset_path, type="dataset").download()
        print(f"✅ Artifact 다운로드 성공. 데이터셋 경로: {dataset_dir}")
        final_data_path = dataset_dir
    except Exception as e:
         print("❌ Artifact가 존재하지 않음. 로컬 데이터 사용.")
         final_data_path = args.data_root
         dataset_artifact_exists = False

    # --- Data ---
    train_loader, val_loader = build_dataloaders(
        root=final_data_path,
        batch_size=args.batch_size,
        augment_minority_only=args.augment_minority_only,
    )

    # --- Training ---
    best_val_loss = float('inf') # best_val_loss 추적
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, head, train_loader, criterion, optimizer, device
        )
        val_loss = validate(model, head, val_loader, criterion, device)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # val_loss가 개선되었을 때만 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"🎉 New best model found! Epoch {epoch}, Val Loss: {val_loss:.4f}")

            ckpt_path = os.path.join(args.save_dir, "best_model.pt") # 파일 이름 고정
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }, ckpt_path)

            model_artifact = wandb.Artifact(name="face-age-checkpoints", type="model")
            model_artifact.add_file(ckpt_path)

            # 'latest'와 'best' 별칭을 모두 추가
            run.log_artifact(model_artifact, aliases=[f"epoch_{epoch}", "latest", "best"])

    print("✅ Training complete!")

    model.eval()


    # --- Create Dataset Artifacts ---
    if not dataset_artifact_exists:
        print("Uploading new dataset artifact...")
        data_artifact = wandb.Artifact(
            name="image_data",
            type="dataset",
        )
        # dataset_dir가 아닌 로컬 경로(args.data_root)를 사용해야 합니다.
        data_artifact.add_dir(args.data_root) 
        run.log_artifact(data_artifact)

    run.finish()


if __name__ == "__main__":
    main()
