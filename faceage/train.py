import os, torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from faceage.utils.seed import set_seed
from faceage.utils.wandb_logger import get_wandb
from faceage.data.datasets import build_dataloaders
from faceage.models.cnn import SimpleCNN
from faceage.models.head import SoftHead
from faceage.losses.soft_label import soft_ce_loss
from faceage.metrics.mae import expected_age_from_logits, mae
from faceage.utils.misc import save_checkpoint

def train_one_epoch(model, head, loader, opt, device, scaler=None):
    model.train(); head.train()
    total = 0.0
    for imgs, soft, race, _age in tqdm(loader, desc="Train", leave=False):
        imgs, soft, race = imgs.to(device), soft.to(device), race.to(device)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                feat = model(imgs)
                logits = head(feat, race)
                loss = soft_ce_loss(logits, soft)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        else:
            feat = model(imgs); logits = head(feat, race); loss = soft_ce_loss(logits, soft)
            loss.backward(); opt.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, head, loader, device):
    model.eval(); head.eval()
    maes = []
    for imgs, _soft, race, age in tqdm(loader, desc="Valid", leave=False):
        imgs, race, age = imgs.to(device), race.to(device), age.to(device)
        feat = model(imgs)
        logits = head(feat, race)
        pred_age = expected_age_from_logits(logits)
        maes.append(mae(pred_age, age))
    return sum(maes) / max(1, len(maes))

@hydra.main(version_base=None, config_path="config", config_name="train/default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = build_dataloaders(cfg)
    featnet = SimpleCNN(in_channels=cfg.model.in_channels).to(device)
    head = SoftHead(in_dim=128, num_bins=cfg.data.num_bins, use_race=cfg.data.use_race_onehot, dropout=cfg.model.dropout).to(device)

    params = list(featnet.parameters()) + list(head.parameters())
    if cfg.train.optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(params, lr=cfg.train.lr)
    else:
        opt = torch.optim.Adam(params, lr=cfg.train.lr)

    wb = get_wandb(cfg)
    scaler = GradScaler() if (cfg.amp and device == "cuda") else None

    best = 1e9
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs+1):
        tl = train_one_epoch(featnet, head, train_loader, opt, device, scaler)
        vm = evaluate(featnet, head, val_loader, device)

        print(f"[Epoch {epoch}] train_loss={tl:.4f} val_mae={vm:.3f}")
        if hasattr(wb, "log"): wb.log({"epoch": epoch, "train/loss": tl, "val/mae": vm, "lr": opt.param_groups[0]["lr"]})

        if vm < best:
            best = vm
            save_checkpoint({
                "feat": featnet.state_dict(),
                "head": head.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "val_mae": vm,
            }, os.path.join(cfg.train.checkpoint_dir, "best.pt"))

    if hasattr(wb, "finish"): wb.finish()
    print(f"Best val MAE: {best:.3f}")

if __name__ == "__main__":
    main()