import os
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import pytorch_lightning as pl
import torchmetrics


def get_dataloaders( data_dir_train: str, data_dir_test: str, batch_size: int = 32, augment: bool = False, resolution: int = 224, val_split: float = 0.2, random_seed: int = 42 ):
    # Transforms
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std =[0.229, 0.224, 0.225])
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std =[0.229, 0.224, 0.225])
        ])

    test_tf = transforms.Compose([
        transforms.Resize(resolution + 32),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
    ])

    # Full training dataset
    train_dataset = datasets.ImageFolder(root=data_dir_train, transform=train_tf)
    targets = [s[1] for s in train_dataset.samples]

    # Stratified split
    splitter = StratifiedShuffleSplit( n_splits=1, test_size=val_split, random_state=random_seed )
    train_idx, val_idx = next(splitter.split(train_dataset.samples, targets))

    train_subset = Subset(train_dataset, train_idx)
    val_subset   = Subset(train_dataset, val_idx)

    train_loader = DataLoader( train_subset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True )
    val_loader = DataLoader( val_subset, batch_size=batch_size, shuffle=False, num_workers=8,  persistent_workers=True  )

    # Test dataset / loader
    test_dataset = datasets.ImageFolder(root=data_dir_test, transform=test_tf)
    test_loader = DataLoader( test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True )

    return train_loader, val_loader, test_loader


class LitModel(pl.LightningModule):
    def __init__(self, model, freeze_strategy="all", unfreeze_epoch=3, head_lr=1e-3, layer4_lr=1e-4, weight_decay=1e-4, label_smoothing=0.0,  optimizer_name="Adam", mixup_alpha=0.0, train_steps=None, total_epochs=None):
        super().__init__()
        self.model = model
        self.freeze_strategy = freeze_strategy
        self.unfreeze_epoch = unfreeze_epoch
        self.layer4_unfrozen = False

        self.head_lr = head_lr
        self.layer4_lr = layer4_lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.optimizer_name = optimizer_name
        self.mixup_alpha = mixup_alpha
        self.train_steps = train_steps
        self.total_epochs = total_epochs

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def mixup_data(self, x, y):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1
        idx = torch.randperm(x.size(0), device=x.device)
        mixed_x = lam * x + (1 - lam) * x[idx]
        return mixed_x, y, y[idx], lam

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.mixup_alpha > 0:
            x, y_a, y_b, lam = self.mixup_data(x, y)
            logits = self(x)
            loss = lam * F.cross_entropy(logits, y_a, label_smoothing=self.label_smoothing) + \
                   (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=self.label_smoothing)
            preds = logits.argmax(dim=1)
            self.train_acc.update(preds, y)
        else:
            logits = self(x)
            loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
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

    def on_epoch_start(self):
        if (self.freeze_strategy == "gradual" and self.current_epoch >= self.unfreeze_epoch and not self.layer4_unfrozen):
            self.print(f"Epoch {self.current_epoch}: unfreezing layer4")
            for p in self.model.model.layer4.parameters():
                p.requires_grad = True
            self.layer4_unfrozen = True

    def configure_optimizers(self):
        if self.optimizer_name == "RMSprop":
            Optim = torch.optim.RMSprop
        elif self.optimizer_name == "SGD":
            Optim = torch.optim.SGD
        else:
            Optim = torch.optim.Adam

        if self.freeze_strategy == "gradual":
            params = [
                {"params": self.model.model.fc.parameters(),     "lr": self.head_lr},
                {"params": self.model.model.layer4.parameters(), "lr": self.layer4_lr}
            ]
        else:
            params = self.model.model.fc.parameters()

        optimizer = Optim(params, weight_decay=self.weight_decay)

        if self.train_steps and self.total_epochs:
            scheduler = torch.optim.lr_scheduler.OneCycleLR( optimizer, max_lr=self.head_lr, epochs=self.total_epochs, steps_per_epoch=self.train_steps )
            return { "optimizer": optimizer, "lr_scheduler": { "scheduler": scheduler, "interval": "step" } }

        return optimizer
