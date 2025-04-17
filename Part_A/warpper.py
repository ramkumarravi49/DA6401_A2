import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

class LitModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.lr = learning_rate
        # torchmetrics auto-handles epoch accumulation
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), on_epoch=True, prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
