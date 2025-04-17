import torch.nn as nn
import torchvision.models as models

class ResNetFinetuner(nn.Module):
    def __init__(self, num_classes=10, freeze_strategy="all", dropout=0.0):
        """
        Args:
            num_classes (int): Number of output classes.
            freeze_strategy (str): Either "all" (freeze all except final layer) or "gradual"
                (initially freeze layer4; it will be unfrozen later).
            dropout (float): Dropout rate before the final fc layer.
        """
        super(ResNetFinetuner, self).__init__()
        self.freeze_strategy = freeze_strategy
        self.dropout = dropout

        # Load the pre-trained ResNet-50 model from torchvision.
        self.model = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer:
        # add a dropout layer before a new linear layer with 10 outputs.
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(num_features, num_classes)
        )
        
        if self.freeze_strategy == "all":
            # Freeze all parameters except those of the final fc layer.
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif self.freeze_strategy == "gradual":
            # Freeze everything initially; unfreeze fc only.
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
            # Note: layer4 remains frozen; it will be unfrozen later via the Lightning module.
        
    def forward(self, x):
        return self.model(x)
