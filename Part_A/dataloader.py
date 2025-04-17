import sys
from pathlib import Path
#sys.path.append(str(Path(__file__).resolve().parent.parent))

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

def get_dataloaders(data_dir, batch_size=32, augment=False, val_split=0.2, random_seed=42):
    if augment:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Perform stratified split into train/validation
    targets = [sample[1] for sample in full_dataset.samples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=random_seed)
    train_idx, val_idx = next(splitter.split(full_dataset.samples, targets))

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
