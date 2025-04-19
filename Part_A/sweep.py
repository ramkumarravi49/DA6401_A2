# sweep.py — updated to use merged helper.py with MixUp and One‑Cycle LR
import wandb, torch, gc
import pytorch_lightning as pl
from pathlib import Path
from model import CNN
from helper import get_dataloaders_with_test, LitModel  # ✅ updated import
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn

# Paths
root_dir = Path(__file__).resolve().parent
data_dir = root_dir / "data"
if not data_dir.exists():
    raise FileNotFoundError(f"{data_dir} not found")

# Activation mapping
activation_map = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish
}

# Sweep Configuration
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "activation_fn":      {"values": ["mish", "gelu"]},
        "dense_neurons":      {"values": [256, 512]},
        "dropout":            {"values": [0.1, 0.2]},
        "filter_organization":{"values": ["hybrid_32_64", "pyramid_256_16"]},
        "learning_rate":      {"min": 1e-5, "max": 5e-4},
        "batch_size":         {"values": [16, 32, 64]},
        "weight_decay":       {"values": [0, 1e-5, 5e-5, 1e-4]},
        "filter_size":        {"values": [3, 5]},
        "data_aug":           {"values": [True]}
    }
}

def sweep_train():
    defaults = {
        "activation_fn": "mish",
        "dense_neurons": 256,
        "dropout": 0.1,
        "filter_organization": "hybrid_32_64",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "weight_decay": 1e-4,
        "filter_size": 3,
        "data_aug": True
    }
    wandb.init(config=defaults)
    config = wandb.config
    wandb.run.name = (
        f"act_{config.activation_fn}_fo_{config.filter_organization}_bs_{config.batch_size}"
        f"_lr_{config.learning_rate:.1e}_wd_{config.weight_decay:.1e}_do_{config.dropout}"
    )

    # Logger & callbacks
    wandb_logger = WandbLogger(entity="cs24m037-iit-madras", project="DA6401_A2")
    early_stop   = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Load data
    train_loader, val_loader, _ = get_dataloaders_with_test(
        data_dir=data_dir,
        batch_size=config.batch_size,
        augment=config.data_aug
    )
    steps_per_epoch = len(train_loader)

    # Model
    model = CNN(
        num_classes=10,
        filter_size=config.filter_size,
        activation_fn=activation_map[config.activation_fn],
        dense_neurons=config.dense_neurons,
        dropout=config.dropout,
        batchnorm=True,
        filter_organization=config.filter_organization
    )

    # LitModel with One-Cycle LR and MixUp
    lit_model = LitModel(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        train_steps=steps_per_epoch,
        epochs=35,
        mixup_alpha=0.4
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=35,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop]
    )
    trainer.fit(lit_model, train_loader, val_loader)

    wandb.finish()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep_config,
        entity="cs24m037-iit-madras",
        project="DA6401_A2"
    )
    wandb.agent(sweep_id, function=sweep_train, entity="cs24m037-iit-madras", project="DA6401_A2")
