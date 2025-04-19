# # b_train.py — Final Version (updated to use helper.py)

# import sys
# from pathlib import Path
# import gc
# import torch
# torch.set_float32_matmul_precision('high')

# from b_model import ResNetFinetuner
# from b_helper import get_dataloaders, LitModel 

# import torchmetrics
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# root_dir = Path(__file__).resolve().parent
# data_dir_train = root_dir / "data" / "train"
# data_dir_test = root_dir / "data" / "Test"
# if not data_dir_train.exists() or not data_dir_test.exists():
#     raise FileNotFoundError("Check data/train and data/Test folders")

# configs = [
#     {
#         "ckpt_name": "top1.ckpt",
#         "freeze": "gradual",
#         "resolution": 299,
#         "batch_size": 64,
#         "dropout": 0.2,
#         "data_aug": True,
#         "head_lr": 0.0031429086775672,
#         "layer4_lr": 0.0001469814980671,
#         "weight_decay": 0.0001051052813562,
#         "label_smoothing": 0.0645085370593245,
#         "unfreeze_epoch": 5,
#         "optimizer": "Adam"
#     },
#     {
#         "ckpt_name": "top2.ckpt",
#         "freeze": "gradual",
#         "resolution": 299,
#         "batch_size": 64,
#         "dropout": 0.2,
#         "data_aug": True,
#         "head_lr": 0.0018686009698193,
#         "layer4_lr": 0.0000214868539084,
#         "weight_decay": 0.0007814767171477,
#         "label_smoothing": 0.0102967888051705,
#         "unfreeze_epoch": 3,
#         "optimizer": "Adam"
#     }
# ]

# if __name__ == "__main__":
#     for idx, cfg in enumerate(configs, start=1):
#         print(f"\n=== Running configuration #{idx} ===")
#         ckpt_path = root_dir / cfg["ckpt_name"]

#         if ckpt_path.exists():
#             print(f"Checkpoint {ckpt_path.name} exists. Loading model...")
#             backbone = ResNetFinetuner(
#                 num_classes=10,
#                 freeze_strategy=cfg["freeze"],
#                 dropout=cfg["dropout"]
#             )
#             lit_model = LitModel.load_from_checkpoint(
#                 str(ckpt_path),
#                 model=backbone,
#                 freeze_strategy=cfg["freeze"],
#                 unfreeze_epoch=cfg["unfreeze_epoch"],
#                 head_lr=cfg["head_lr"],
#                 layer4_lr=cfg["layer4_lr"],
#                 weight_decay=cfg["weight_decay"],
#                 label_smoothing=cfg["label_smoothing"],
#                 optimizer_name=cfg["optimizer"]
#             )
#             lit_model = lit_model.to(device)

#         else:
#             print(f"No checkpoint found. Training new model and saving to {ckpt_path.name}...")

#             train_loader, test_loader = get_dataloaders(
#                 data_dir_train=str(data_dir_train),
#                 data_dir_test=str(data_dir_test),
#                 batch_size=cfg["batch_size"],
#                 augment=cfg["data_aug"],
#                 resolution=cfg["resolution"]
#             )

#             backbone = ResNetFinetuner(
#                 num_classes=10,
#                 freeze_strategy=cfg["freeze"],
#                 dropout=cfg["dropout"]
#             )
#             lit_model = LitModel(
#                 model=backbone,
#                 freeze_strategy=cfg["freeze"],
#                 unfreeze_epoch=cfg["unfreeze_epoch"],
#                 head_lr=cfg["head_lr"],
#                 layer4_lr=cfg["layer4_lr"],
#                 weight_decay=cfg["weight_decay"],
#                 label_smoothing=cfg["label_smoothing"],
#                 optimizer_name=cfg["optimizer"]
#             ).to(device)

#             checkpoint = ModelCheckpoint(
#                 dirpath=root_dir,
#                 filename=cfg["ckpt_name"].replace(".ckpt", ""),
#                 save_top_k=1,
#                 save_last=False,
#                 monitor="val_acc",
#                 mode="max"
#             )

#             trainer = pl.Trainer(
#                 max_epochs=17,
#                 accelerator="gpu" if torch.cuda.is_available() else "cpu",
#                 devices=1,
#                 logger=False,
#                 callbacks=[checkpoint]
#             )

#             trainer.fit(lit_model, train_loader, test_loader)

#         # Evaluate on test set
#         print("Evaluating on test set...")
#         _, test_loader = get_dataloaders(
#             data_dir_train=str(data_dir_train),
#             data_dir_test=str(data_dir_test),
#             batch_size=cfg["batch_size"],
#             augment=cfg["data_aug"],
#             resolution=cfg["resolution"]
#         )

#         lit_model = lit_model.to(device)
#         lit_model.eval()
#         metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

#         with torch.no_grad():
#             for x, y in test_loader:
#                 x, y = x.to(device), y.to(device)
#                 logits = lit_model(x)
#                 preds = torch.argmax(logits, dim=1)
#                 metric.update(preds, y)
#         test_acc = metric.compute().item()
#         print(f"✅ Config #{idx} → Test Accuracy: {test_acc:.4f}")

#         torch.cuda.empty_cache()
#         gc.collect()


# b_train.py — Updated with MixUp and One‑Cycle LR support

import sys
from pathlib import Path
import numpy as np

import gc
import torch
torch.set_float32_matmul_precision('high')

from b_model import ResNetFinetuner
from b_helper import get_dataloaders, LitModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = Path(__file__).resolve().parent
data_dir_train = root_dir / "data" / "train"
data_dir_test = root_dir / "data" / "Test"
if not data_dir_train.exists() or not data_dir_test.exists():
    raise FileNotFoundError("Check data/train and data/Test folders")

# Define model configurations
configs = [
    {
        "ckpt_name": "top1.ckpt",
        "freeze": "gradual",
        "resolution": 299,
        "batch_size": 64,
        "dropout": 0.2,
        "data_aug": True,
        "head_lr": 0.0031429086775672,
        "layer4_lr": 0.0001469814980671,
        "weight_decay": 0.0001051052813562,
        "label_smoothing": 0.0645085370593245,
        "unfreeze_epoch": 5,
        "optimizer": "Adam",
        "mixup_alpha": 0.4,
        "max_epochs": 17
    },
    {
        "ckpt_name": "top2.ckpt",
        "freeze": "gradual",
        "resolution": 299,
        "batch_size": 64,
        "dropout": 0.2,
        "data_aug": True,
        "head_lr": 0.0018686009698193,
        "layer4_lr": 0.0000214868539084,
        "weight_decay": 0.0007814767171477,
        "label_smoothing": 0.0102967888051705,
        "unfreeze_epoch": 3,
        "optimizer": "Adam",
        "mixup_alpha": 0.4,
        "max_epochs": 17
    }
]

if __name__ == "__main__":
    for idx, cfg in enumerate(configs, start=1):
        print(f"\n=== Running configuration #{idx} ===")
        ckpt_path = root_dir / cfg["ckpt_name"]

        # Load or Train
        if ckpt_path.exists():
            print(f"Checkpoint {ckpt_path.name} exists. Loading model...")
            backbone = ResNetFinetuner(
                num_classes=10,
                freeze_strategy=cfg["freeze"],
                dropout=cfg["dropout"]
            )
            lit_model = LitModel.load_from_checkpoint(
                str(ckpt_path),
                model=backbone,
                freeze_strategy=cfg["freeze"],
                unfreeze_epoch=cfg["unfreeze_epoch"],
                head_lr=cfg["head_lr"],
                layer4_lr=cfg["layer4_lr"],
                weight_decay=cfg["weight_decay"],
                label_smoothing=cfg["label_smoothing"],
                optimizer_name=cfg["optimizer"],
                mixup_alpha=cfg["mixup_alpha"],
                train_steps=None,  # These aren’t needed for resume
                total_epochs=None
            )
            lit_model = lit_model.to(device)

        else:
            print(f"No checkpoint found. Training new model and saving to {ckpt_path.name}...")

            train_loader, test_loader = get_dataloaders(
                data_dir_train=str(data_dir_train),
                data_dir_test=str(data_dir_test),
                batch_size=cfg["batch_size"],
                augment=cfg["data_aug"],
                resolution=cfg["resolution"]
            )

            train_steps = len(train_loader)

            backbone = ResNetFinetuner(
                num_classes=10,
                freeze_strategy=cfg["freeze"],
                dropout=cfg["dropout"]
            )

            lit_model = LitModel(
                model=backbone,
                freeze_strategy=cfg["freeze"],
                unfreeze_epoch=cfg["unfreeze_epoch"],
                head_lr=cfg["head_lr"],
                layer4_lr=cfg["layer4_lr"],
                weight_decay=cfg["weight_decay"],
                label_smoothing=cfg["label_smoothing"],
                optimizer_name=cfg["optimizer"],
                mixup_alpha=cfg["mixup_alpha"],
                train_steps=train_steps,
                total_epochs=cfg["max_epochs"]
            ).to(device)

            checkpoint = ModelCheckpoint(
                dirpath=root_dir,
                filename=cfg["ckpt_name"].replace(".ckpt", ""),
                save_top_k=1,
                save_last=False,
                monitor="val_acc",
                mode="max"
            )

            trainer = pl.Trainer(
                max_epochs=cfg["max_epochs"],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                logger=False,
                callbacks=[checkpoint]
            )

            trainer.fit(lit_model, train_loader, test_loader)

        # Evaluate on test set
        print("Evaluating on test set...")
        _, test_loader = get_dataloaders(
            data_dir_train=str(data_dir_train),
            data_dir_test=str(data_dir_test),
            batch_size=cfg["batch_size"],
            augment=cfg["data_aug"],
            resolution=cfg["resolution"]
        )

        lit_model = lit_model.to(device)
        lit_model.eval()

        from torchmetrics import Accuracy
        metric = Accuracy(task="multiclass", num_classes=10).to(device)

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = lit_model(x)
                preds = torch.argmax(logits, dim=1)
                metric.update(preds, y)
        test_acc = metric.compute().item()
        print(f"✅ Config #{idx} → Test Accuracy: {test_acc:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

