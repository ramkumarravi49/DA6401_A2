import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

class LitModel(pl.LightningModule):
    def __init__(self, model, freeze_strategy="all", unfreeze_epoch=3):
        """
        Args:
            model (nn.Module): The ResNetFinetuner model.
            freeze_strategy (str): "all" or "gradual".
            unfreeze_epoch (int): The epoch at which to unfreeze layer4 (only if using gradual unfreezing).
        """
        super().__init__()
        self.model = model
        self.freeze_strategy = freeze_strategy
        self.unfreeze_epoch = unfreeze_epoch
        self.layer4_unfrozen = False  # flag to prevent multiple unfreeze calls
        
        # Metrics for multiclass classification (10 classes)
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_acc.reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()
        
    def configure_optimizers(self):
        if self.freeze_strategy == "gradual":
            # Set up two parameter groups: one for the fc and one for layer4.
            # (Layer4 is still frozen initially; its parameters will not update until unfrozen.)
            optimizer = torch.optim.Adam([
                {"params": self.model.model.fc.parameters(), "lr": 1e-3},
                {"params": self.model.model.layer4.parameters(), "lr": 1e-4}
            ], weight_decay=1e-4)
        else:
            optimizer = torch.optim.Adam(
                self.model.model.fc.parameters(), lr=1e-3, weight_decay=1e-4
            )
        return optimizer
    
    def on_epoch_start(self):
        # For gradual unfreezing: if current_epoch >= unfreeze_epoch, unfreeze layer4.
        if self.freeze_strategy == "gradual" and (self.current_epoch >= self.unfreeze_epoch) and (not self.layer4_unfrozen):
            self.print(f"Epoch {self.current_epoch}: Unfreezing layer4.")
            for param in self.model.model.layer4.parameters():
                param.requires_grad = True
            self.layer4_unfrozen = True
