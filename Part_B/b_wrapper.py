import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

class LitModel(pl.LightningModule):
    def __init__(self,
                 model,
                 freeze_strategy="all",
                 unfreeze_epoch=3,
                 head_lr=1e-3,
                 layer4_lr=1e-4,
                 weight_decay=1e-4,
                 label_smoothing=0.0,
                 optimizer_name="Adam"):
        super().__init__()
        self.model = model
        self.freeze_strategy = freeze_strategy
        self.unfreeze_epoch = unfreeze_epoch
        self.layer4_unfrozen = False

        # Hyperparams
        self.head_lr = head_lr
        self.layer4_lr = layer4_lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.optimizer_name = optimizer_name

        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()

    def on_epoch_start(self):
        # Gradual unfreeze of layer4
        if ( self.freeze_strategy == "gradual"
         and self.current_epoch >= self.unfreeze_epoch
         and not self.layer4_unfrozen ):
            self.print(f"Epoch {self.current_epoch}: unfreeezing layer4")
            for p in self.model.model.layer4.parameters():
                p.requires_grad = True
            self.layer4_unfrozen = True

    def configure_optimizers(self):
        # Choose optimizer class
        if self.optimizer_name == "RMSprop":
            Optim = torch.optim.RMSprop
        elif self.optimizer_name == "SGD":
            Optim = torch.optim.SGD
        else:
            Optim = torch.optim.Adam

        if self.freeze_strategy == "gradual":
            optimizer = Optim([
                {"params": self.model.model.fc.parameters(),     "lr": self.head_lr},
                {"params": self.model.model.layer4.parameters(), "lr": self.layer4_lr}
            ], weight_decay=self.weight_decay)
        else:
            optimizer = Optim(
                self.model.model.fc.parameters(),
                lr=self.head_lr,
                weight_decay=self.weight_decay
            )
        return optimizer
