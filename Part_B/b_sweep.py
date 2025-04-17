import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
data_dir_train = root_dir / "data" / "train"
data_dir_test = root_dir / "data" / "Test"

if not data_dir_train.exists() or not data_dir_test.exists():
    raise FileNotFoundError("Data directories not found. Please check the paths.")

import wandb
import torch
import gc
import pytorch_lightning as pl
from b_model import ResNetFinetuner
from b_dataloader import get_dataloaders
from b_wrapper import LitModel
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Optionally alias ModelCheckpoint as LitModelCheckpoint
LitModelCheckpoint = ModelCheckpoint

sweep_config = {
    "method": "grid",
    "metric": {
        "name": "val_acc",
        "goal": "maximize"
    },
    "parameters": {
        "freeze": {"values": ["all", "gradual"]},
        "resolution": {"values": [224, 299]},
        "batch_size": {"values": [32, 64, 128]},
        "dropout": {"values": [0, 0.2, 0.4]},
        "data_aug": {"values": [True]}
    }
}

def sweep_train():
    # Set default hyperparameter values.
    config_defaults = {
        "freeze": "gradual",
        "resolution": 224,
        "batch_size": 64,
        "dropout": 0.2,
        "data_aug": True
    }
    wandb.init(config=config_defaults)
    config = wandb.config
    # Create a descriptive run name.
    wandb.run.name = (
        f"freeze_{config.freeze}_res_{config.resolution}_bs_{config.batch_size}_"
        f"drop_{config.dropout}_aug_{config.data_aug}"
    )
    
    # Initialize the WandB logger with the specified entity and project.
    wandb_logger = WandbLogger(project="DA6401_A2", entity="cs24m037-iit-madras", log_model=False)
    
    # Get the data loaders.
    train_loader, val_loader = get_dataloaders(
        data_dir_train=str(data_dir_train),
        data_dir_test=str(data_dir_test),
        batch_size=config.batch_size,
        augment=config.data_aug,
        resolution=config.resolution
    )
    
    # Instantiate the ResNetFinetuner model with specified dropout and freeze strategy.
    model = ResNetFinetuner(num_classes=10, freeze_strategy=config.freeze, dropout=config.dropout)
    
    # Wrap the model with our Lightning module.
    # For gradual unfreezing, we set unfreeze_epoch to 3 (adjust as needed).
    lit_model = LitModel(model, freeze_strategy=config.freeze, unfreeze_epoch=3)
    
    # Early stopping callback.
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    
    # Checkpoint callback using LitModelCheckpoint for seamless uploading.
    checkpoint_callback = LitModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True
    )
    
    # Create the Trainer.
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        accelerator="gpu",
        callbacks=[early_stop, checkpoint_callback],
        devices=1
    )
    
    # Start training.
    trainer.fit(lit_model, train_loader, val_loader)
    wandb.finish()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="DA6401_A2", entity="cs24m037-iit-madras")
    wandb.agent(sweep_id, function=sweep_train, entity="cs24m037-iit-madras", project="DA6401_A2")
