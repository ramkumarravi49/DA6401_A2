# traintop3.py — logs test accuracy and grid to WandB
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pathlib import Path
import wandb
from model import CNN
from helper import get_dataloaders_with_test
from helper import LitModel
from pytorch_lightning.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch.nn.functional as F
import random

# Map string to activation function
activation_map = {
    "relu": nn.ReLU, "gelu": nn.GELU,
    "silu": nn.SiLU, "mish": nn.Mish
}

# Top config (others commented)
top3_configs = [
    {"activation_fn": "mish", "filter_organization": "32_512", "data_aug": True, "batchnorm": True,
     "dropout": 0.4, "dense_neurons": 256, "learning_rate": 0.000177, "filter_size": 3, "batch_size": 64}
    # Other two configs omitted on purpose
]

max_epochs = 25
root_dir = Path(__file__).resolve().parent
data_dir = root_dir / "data"

def show_predictions_grid(model, dataset, class_labels, device, k=3):
    """
    10×3 grid with:
    - True & predicted labels (top)
    - Top‑k scores (top, below labels)
    - Image at bottom
    - Black line after each class row
    """
    model.eval()
    samples = {c: [] for c in range(len(class_labels))}
    
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    for idx in idxs:
        img, lbl = dataset[idx]
        if len(samples[lbl]) >= 3:
            continue
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device)).squeeze()
            probs = F.softmax(out, dim=0)
            pred = probs.argmax().item()
            topv, topi = probs.topk(k)
            samples[lbl].append((img, lbl, pred, topi.cpu(), topv.cpu()))
        if all(len(v) == 3 for v in samples.values()):
            break
    
    fig, axs = plt.subplots(10, 3, figsize=(11, 32))
    fig.suptitle("Prediction Grid with Top‑K Confidence Scores", fontsize=16)
    fig.subplots_adjust(top=0.94, bottom=0.05, hspace=1.3)  # More vertical space
    
    for r, items in enumerate(samples.values()):
        for c, (img, true, pred, topi, topv) in enumerate(items):
            ax = axs[r, c]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis("off")
            
            # True/Pred label on top
            label_color = "green" if true == pred else "red"
            ax.text(0.5, 1.35, f"True: {class_labels[true]}", transform=ax.transAxes,
                   fontsize=9, va='center', ha='center', color='black',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2'))
            
            ax.text(0.5, 1.20, f"Pred: {class_labels[pred]}", transform=ax.transAxes,
                   fontsize=9, va='center', ha='center', color=label_color,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.2'))
            
            # Top-k confidence scores in a single text box
            conf_text = "\n".join([f"{class_labels[i]}: {v:.2f}" for i, v in zip(topi, topv)])
            ax.text(0.5, 1.00, conf_text, transform=ax.transAxes,
                   fontsize=7, va='top', ha='center', color='blue',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightblue', 
                           boxstyle='round,pad=0.3'))
    
    # Draw horizontal line after every row
    for r in range(10):
        ax0 = axs[r, 0]
        y = ax0.get_position().y0 - 0.01
        line = Line2D([0, 1], [y, y], transform=fig.transFigure,
                     color='black', linewidth=2)
        fig.add_artist(line)
    
    return fig



# Train and evaluate best config
for i, cfg in enumerate(top3_configs, 1):
    print(f"\n=== Model {i}: {cfg}")

    # 1) Prepare data + compute steps_per_epoch
    train_loader, val_loader, test_loader = get_dataloaders_with_test(
        data_dir=data_dir,
        batch_size=cfg["batch_size"],
        augment=cfg["data_aug"]
    )
    steps_per_epoch = len(train_loader)

    # 2) Build the CNN + LightningModule
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

    ckpt_path = root_dir / "best_model.ckpt"
    if ckpt_path.exists():
        # ─── Skip training, just load weights ───────────────────────────────────────
        print(f"🔄 Checkpoint found at {ckpt_path}, loading weights and skipping training…")
        ckpt = torch.load(str(ckpt_path), map_location=lit.device)
        # load the full LightningModule state dict (keys include 'model.*')
        lit.load_state_dict(ckpt["state_dict"])
        print(f"🔄 Loaded checkpoint from {ckpt_path}, skipping training.")
    else:
        # ─── No checkpoint: train and then save ────────────────────────────────────
        early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[early_stop],
            logger=False
        )
        trainer.fit(lit, train_loader, val_loader)

        # Save for next time
        trainer.save_checkpoint(str(ckpt_path))
        print(f"Model checkpoint saved to: {ckpt_path}")

    # 3) Evaluate on test set (same as before)
    test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(lit.device)
    lit.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(lit.device), y.to(lit.device)
            preds = lit(x).argmax(dim=1)
            test_acc.update(preds, y)
    final_acc = test_acc.compute()
    print(f"✅ Model {i} Test Accuracy: {final_acc:.4f}")

    # 4) Visualize & log (same as before)
    if i == 1:
        test_dataset = test_loader.dataset
        fig = show_predictions_grid(lit, test_dataset, test_dataset.classes, lit.device)

        # Save the grid locally
        save_path = root_dir / "final_prediction.png"
        fig.savefig(save_path, dpi=300)
        print(f"📸 Saved grid at: {save_path}")

        wandb.init(
            project="DA6401_A2",
            entity="cs24m037-iit-madras",
            name="BestModel-TestGrid",
            config=cfg,
            reinit=True
        )
        wandb.log({
            "Test Accuracy": final_acc,
            "Prediction Grid": wandb.Image(fig)
        })
        wandb.finish()