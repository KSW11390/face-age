import torch, os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from faceage.data.datasets import build_dataloaders
from faceage.models.cnn import SimpleCNN
from faceage.models.head import SoftHead
from faceage.metrics.mae import expected_age_from_logits, mae
from faceage.utils.misc import load_checkpoint

@hydra.main(version_base=None, config_path="config", config_name="train/default")
def main(cfg: DictConfig):
    ckpt_path = os.path.join(cfg.train.checkpoint_dir, "best.pt")
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    featnet = SimpleCNN(in_channels=cfg.model.in_channels).to(device)
    head = SoftHead(in_dim=128, num_bins=cfg.data.num_bins, use_race=cfg.data.use_race_onehot, dropout=cfg.model.dropout).to(device)
    featnet.load_state_dict(ckpt["feat"]); head.load_state_dict(ckpt["head"])

    _, val_loader = build_dataloaders(cfg)
    featnet.eval(); head.eval()
    maes=[]
    with torch.no_grad():
        for imgs, _soft, race, age in tqdm(val_loader, desc="Eval"):
            imgs, race, age = imgs.to(device), race.to(device), age.to(device)
            logits = head(featnet(imgs), race)
            pred_age = expected_age_from_logits(logits)
            maes.append(mae(pred_age, age))
    print(f"Eval MAE: {sum(maes)/len(maes):.3f}")

if __name__ == "__main__":
    main()