class NullLogger:
    def __init__(self): pass
    def log(self, *a, **kw): pass
    def finish(self): pass

def get_wandb(cfg):
    if not cfg.train.wandb.enable:
        return NullLogger()
    import wandb
    wandb.init(project=cfg.train.wandb.project, name=cfg.train.wandb.name)
    wandb.config.update(dict(cfg))
    return wandb