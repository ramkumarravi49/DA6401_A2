import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir_train, data_dir_test, batch_size=32, augment=False, resolution=224):
    """
    Args:
        data_dir_train (str): Path to the training folder.
        data_dir_test (str): Path to the test/validation folder.
        batch_size (int): Batch size.
        augment (bool): If True, apply data augmentation on the training set.
        resolution (int): The resolution (both width and height) to use.
        
    Returns:
        (train_loader, test_loader): PyTorch DataLoaders for train and test sets.
    """
    if augment:
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
    test_transforms = transforms.Compose([
        transforms.Resize(resolution + 32),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=data_dir_train, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=data_dir_test, transform=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader
