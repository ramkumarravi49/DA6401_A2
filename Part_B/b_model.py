# b_model.py (unchanged)
import torch.nn as nn
import torchvision.models as models

class ResNetFinetuner(nn.Module):
    def __init__(self, num_classes=10,
                 freeze_strategy="all", dropout=0.0):
        super().__init__()
        self.freeze_strategy = freeze_strategy
        self.model = models.resnet50(pretrained=True)

        # Replace final fc
        n_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, num_classes)
        )

        # Apply initial freezing
        if freeze_strategy == "all":
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.fc.parameters():
                p.requires_grad = True
        elif freeze_strategy == "gradual":
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.fc.parameters():
                p.requires_grad = True
        # layer4 stays frozen until the wrapper unfreezes it

    def forward(self, x):
        return self.model(x)
