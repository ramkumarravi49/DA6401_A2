# warpper.py    ← **modified** to add MixUp & One‑Cycle LR
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import numpy as np

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
        # use original labels for accuracy metric
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
