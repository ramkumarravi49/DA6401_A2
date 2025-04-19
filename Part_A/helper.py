# helper.py — merged dataloader and wrapper with MixUp and One‑Cycle LR
import torch
import torch.nn.functional as F
import torchmetrics
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
import pytorch_lightning as pl

# Dataloader with test split
def get_dataloaders_with_test(data_dir, batch_size=32, augment=False, val_split=0.2, random_seed=42):
    if augment:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])  
    
    data_dir = Path(data_dir)
    train_path = data_dir / "train"
    test_path = data_dir / "test"

    full_dataset = datasets.ImageFolder(train_path, transform=transform)
    targets = [sample[1] for sample in full_dataset.samples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=random_seed)
    train_idx, val_idx = next(splitter.split(full_dataset.samples, targets))

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size, shuffle=False)

    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Lightning Module with MixUp and One‑Cycle LR
class LitModel(pl.LightningModule):
    def __init__(
        self, model, learning_rate=1e-3, weight_decay=1e-4,
        train_steps=None, epochs=None, mixup_alpha=0.4
    ):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        self.wd = weight_decay
        self.train_steps = train_steps
        self.epochs = epochs
        self.mixup_alpha = mixup_alpha

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def mixup_data(self, x, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
        batch_size = x.size(0)
        idx = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[idx]
        return mixed_x, y, y[idx], lam

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_mixed, y_a, y_b, lam = self.mixup_data(x, y)
        logits = self(x_mixed)
        loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)

        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lr,
            epochs=self.epochs,
            steps_per_epoch=self.train_steps
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"}
        }
