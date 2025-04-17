# traintop3.py    ← **modified** to pass train_steps, epochs & alpha
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pathlib import Path
from model import CNN
from dataloader import get_dataloaders_with_test
from warpper import LitModel
from pytorch_lightning.callbacks import EarlyStopping

activation_map = {
    "relu": nn.ReLU, "gelu": nn.GELU,
    "silu": nn.SiLU, "mish": nn.Mish
}

top3_configs = [
    {"activation_fn":"mish","filter_organization":"32_512","data_aug":True,"batchnorm":True,
     "dropout":0.4,"dense_neurons":256,"learning_rate":0.000177,"filter_size":3,"batch_size":64}
    # {"activation_fn":"mish","filter_organization":"hybrid_16_256_64","data_aug":True,"batchnorm":True,
    #  "dropout":0.3,"dense_neurons":256,"learning_rate":0.000339,"filter_size":5,"batch_size":64},
    # {"activation_fn":"mish","filter_organization":"hybrid_32_64","data_aug":True,"batchnorm":True,
    #  "dropout":0.2,"dense_neurons":512,"learning_rate":0.000244,"filter_size":5,"batch_size":64}
]

max_epochs = 25
root_dir = Path(__file__).resolve().parent
data_dir = root_dir / "data"

for i, cfg in enumerate(top3_configs, 1):
    print(f"\n=== Model {i}: {cfg}")

    train_loader, val_loader, test_loader = get_dataloaders_with_test(
        data_dir=data_dir,
        batch_size=cfg["batch_size"],
        augment=cfg["data_aug"]
    )

    steps_per_epoch = len(train_loader)
    model = CNN(
        num_classes=10,
        filter_size=cfg["filter_size"],
        activation_fn=activation_map[cfg["activation_fn"]],
        dense_neurons=cfg["dense_neurons"],
        dropout=cfg["dropout"],
        batchnorm=cfg["batchnorm"],
        filter_organization=cfg["filter_organization"]
    )

    lit = LitModel(
        model,
        learning_rate=cfg["learning_rate"],
        weight_decay=1e-4,
        train_steps=steps_per_epoch,
        epochs=max_epochs,
        mixup_alpha=0.4
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[early_stop],
        logger=False
    )

    trainer.fit(lit, train_loader, val_loader)

    # test‐time accuracy
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(lit.device)
    lit.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(lit.device), y.to(lit.device)
            preds = lit(x).argmax(dim=1)
            test_acc.update(preds, y)
    print(f"✅ Model {i} Test Accuracy: {test_acc.compute():.4f}")
