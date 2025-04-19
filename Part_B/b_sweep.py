# b_sweep.py — updated to use b_helper.py (merged dataloader + wrapper)

import sys
from pathlib import Path
import wandb
import torch
import gc
import pytorch_lightning as pl
from b_model   import ResNetFinetuner
from b_helper  import get_dataloaders, LitModel  # ✅ updated imports
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers  import WandbLogger

# Paths
root_dir        = Path(__file__).resolve().parent
data_dir_train  = root_dir / "data" / "train"
data_dir_test   = root_dir / "data" / "Test"
if not data_dir_train.exists() or not data_dir_test.exists():
    raise FileNotFoundError("Check data/train and data/Test folders")

# Fixed sweep constants
fixed_defaults = {
    "freeze":      "gradual",
    "resolution":  299,
    "batch_size":  64,
    "dropout":     0.2,
    "data_aug":    True
}

# Sweep config
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "head_lr":        {"distribution": "uniform", "min": 1e-4,  "max": 5e-3},
        "layer4_lr":      {"distribution": "uniform", "min": 1e-5,  "max": 2e-4},
        "weight_decay":   {"distribution": "uniform", "min": 1e-5,  "max": 5e-4},
        "unfreeze_epoch": {"values":       [4, 5]},
        "label_smoothing":{"distribution": "uniform", "min": 0.04,  "max": 0.08},
        "optimizer":      {"values":       ["Adam", "AdamW"]}
    }
}

def sweep_train():
    config_defaults = {
        # locked‑in constants
        "freeze":           "gradual",
        "resolution":       299,
        "batch_size":       64,
        "dropout":          0.2,
        "data_aug":         True,
        # new tuned hyper‑params (defaulted to midpoint-ish)
        "head_lr":          3e-3,
        "layer4_lr":        1e-4,
        "weight_decay":     2e-4,
        "label_smoothing":  0.06,
        "unfreeze_epoch":   4,
        "optimizer":        "AdamW"
    }

    wandb.init(config=config_defaults)
    config = wandb.config

    # Load data loaders
    train_loader, val_loader, _ = get_dataloaders(
        data_dir_train=str(data_dir_train),
        data_dir_test=str(data_dir_test),
        batch_size=config.batch_size,
        augment=config.data_aug,
        resolution=config.resolution
    )
    train_steps = len(train_loader)

    # Backbone model
    backbone = ResNetFinetuner(
        num_classes=10,
        freeze_strategy=config.freeze,
        dropout=config.dropout
    )

    # Lightning model
    lit_model = LitModel(
        model=backbone,
        freeze_strategy=config.freeze,
        unfreeze_epoch=config.unfreeze_epoch,
        head_lr=config.head_lr,
        layer4_lr=config.layer4_lr,
        weight_decay=config.weight_decay,
        label_smoothing=config.label_smoothing,
        optimizer_name=config.optimizer,
        mixup_alpha=0.4,
        train_steps=train_steps,
        total_epochs=20
    )

    wandb_logger = WandbLogger(
        project="DA6401_A2",
        entity="cs24m037-iit-madras",
        log_model=False
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=4, mode="min")
    checkpoint = ModelCheckpoint(
        monitor="val_acc", mode="max",
        filename="best-{epoch:02d}-{val_acc:.3f}",
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint]
    )

    trainer.fit(lit_model, train_loader, val_loader)
    wandb.finish()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    sweep_id = "a9xddx6w"  # Replace with your sweep ID if needed
    wandb.agent(
        sweep_id,
        function=sweep_train,
        project="DA6401_A2",
        entity="cs24m037-iit-madras"
    )
