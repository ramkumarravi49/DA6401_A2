# b_sweep.py

import sys
from pathlib import Path
import wandb
import torch
import gc
import pytorch_lightning as pl
from b_model      import ResNetFinetuner
from b_dataloader import get_dataloaders
from b_wrapper    import LitModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers  import WandbLogger

root_dir       = Path(__file__).resolve().parent
data_dir_train = root_dir / "data" / "train"
data_dir_test  = root_dir / "data" / "Test"
if not data_dir_train.exists() or not data_dir_test.exists():
    raise FileNotFoundError("Check data/train and data/Test folders")

# Fixed best-known constants
fixed_defaults = {
    "freeze":      "gradual",
    "resolution":  299,
    "batch_size":  64,
    "dropout":     0.2,
    "data_aug":    True
}

# New hyperparameters to sweep (corrected distributions)
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "head_lr":        {"distribution": "log_uniform", "min": 1e-4,  "max": 5e-3},
        "layer4_lr":      {"distribution": "log_uniform", "min": 1e-5,  "max": 5e-4},
        "weight_decay":   {"distribution": "uniform",     "min": 0.0,   "max": 1e-3},
        "unfreeze_epoch": {"values":       [2, 3, 4, 5]},
        "label_smoothing":{"distribution": "uniform",     "min": 0.0,   "max": 0.1},
        "optimizer":      {"values":       ["Adam", "AdamW", "RMSprop"]}
    }
}

def sweep_train():
    wandb.init(config={**fixed_defaults})
    config = wandb.config

    train_loader, val_loader = get_dataloaders(
        data_dir_train=str(data_dir_train),
        data_dir_test =str(data_dir_test),
        batch_size   =config.batch_size,
        augment      =config.data_aug,
        resolution   =config.resolution
    )

    backbone = ResNetFinetuner(
        num_classes     =10,
        freeze_strategy =config.freeze,
        dropout         =config.dropout
    )
    lit_model = LitModel(
        backbone,
        freeze_strategy =config.freeze,
        unfreeze_epoch  =config.unfreeze_epoch,
        head_lr         =config.head_lr,
        layer4_lr       =config.layer4_lr,
        weight_decay    =config.weight_decay,
        label_smoothing =config.label_smoothing,
        optimizer_name  =config.optimizer
    )

    wandb_logger = WandbLogger(
        project="DA6401_A2",
        entity ="cs24m037-iit-madras",
        log_model=False
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=4, mode="min")
    checkpoint = ModelCheckpoint(
        monitor="val_acc", mode="max",
        filename="best-{epoch:02d}-{val_acc:.3f}",
        save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=20,               # now 20 epochs
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
    # Attach to the existing sweep rather than creating a new one
    sweep_id = "a9xddx6w"
    wandb.agent(
        sweep_id,
        function=sweep_train,
        project="DA6401_A2",
        entity="cs24m037-iit-madras"
    )