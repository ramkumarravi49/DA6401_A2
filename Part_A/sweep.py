import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
data_dir = root_dir / "data" / "train"

if not data_dir.exists():
    raise FileNotFoundError(f"Data directory {data_dir} does not exist. Please check the path.")

import os
import wandb
import torch
import gc
import torch.nn as nn
import pytorch_lightning as pl
from model import CNN
from dataloader import get_dataloaders
from warpper import LitModel
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Activation function mapping
activation_map = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish
}

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_acc",
        "goal": "maximize"
    },
    "parameters": {
        "activation_fn": {
            "values": ["relu", "gelu", "silu", "mish"]
        },
        "dense_neurons": {
            "values": [128, 256, 512]
        },
        "dropout": {
            "values": [0.2, 0.3, 0.4]
        },
        "batchnorm": {
            "values": [True]
        },
        "filter_organization": {
            "values": ["16_256", "32_512", "pyramid_256_16", "hybrid_32_64", "hybrid_16_256_64"]
        },
        "learning_rate": { 
            "min": 1e-5,  
            "max": 1e-3   
        },
        "data_aug": {
            "values": [True]
        },
        "filter_size": {
            "values": [3, 5]
        },
        "batch_size": {
            "values": [64, 128]
        }
    }
}

def sweep_train():
    # Set default values for hyperparameters here
    config_defaults = {
        "activation_fn": "mish",
        "dense_neurons": 256,
        "dropout": 0.4,
        "batchnorm": True,
        "filter_organization": "hybrid_32_64",
        "learning_rate": 0.0005,
        "data_aug": True,
        "filter_size": 3,
        "batch_size": 64
    }
    wandb.init(config=config_defaults)  # Initialize wandb at the beginning
    config = wandb.config  # Get configuration parameters
    
    # Build the run name with all hyperparameters
    wandb.run.name = (
        f"act_{config.activation_fn}_"
        f"filtersOrg_{config.filter_organization}_aug_{config.data_aug}_"
        f"bn_{config.batchnorm}_do_{config.dropout}_"
        f"hs_{config.dense_neurons}_lr_{config.learning_rate}_"
        f"filterSize_{config.filter_size}_bs_{config.batch_size}"
    )

    # Initialize WandB logger
    wandb_logger = WandbLogger(entity="cs24m037-iit-madras", project="DA6401_A2", log_model=False)
    
    # Get data loaders with the specified batch size
    train_loader, val_loader = get_dataloaders(
        data_dir=str(data_dir), 
        batch_size=config.batch_size, 
        augment=config.data_aug, 
        val_split=0.2
    )

    # Create the CNN model with the specified hyperparameters
    model = CNN(
        num_classes=10,
        filter_size=config.filter_size,
        activation_fn=activation_map[config.activation_fn],
        dense_neurons=config.dense_neurons,
        dropout=config.dropout,
        batchnorm=config.batchnorm,
        filter_organization=config.filter_organization
    )

    # Wrap the model with the Lightning module
    lit_model = LitModel(model, learning_rate=config.learning_rate)
    
    # Set up early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    # Initialize the trainer with early stopping
    trainer = pl.Trainer(
        max_epochs=20,
        logger=wandb_logger,
        accelerator="gpu",
        callbacks=[early_stop],
        devices=1
    )

    # Train the model
    trainer.fit(lit_model, train_loader, val_loader)
    wandb.finish()
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    # Resume an existing sweep with ID "yitratc9"
    sweep_id = "yitratc9"
    wandb.agent(sweep_id, function=sweep_train, entity="cs24m037-iit-madras", project="DA6401_A2")

    
