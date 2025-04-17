import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10, filter_size=3, activation_fn=nn.ReLU, dense_neurons=128,
                 dropout=0.2, batchnorm=False, filter_organization='16_256'):
        super(CNN, self).__init__()
        layers = []
        in_channels = 3

        # Predefined filter organization
        filter_counts_map = {
            "16_256": [16, 32, 64, 128, 256],
            "32_512": [32, 64, 128, 256, 512],
            "constant_32": [32, 32, 32, 32, 32],
            "pyramid_256_16": [256, 128, 64, 32, 16],
            "64_1024": [64, 128, 256, 512, 1024],
            "pyramid_128_8": [128, 64, 32, 16, 8],
            "pyramid_512_32": [512, 256, 128, 64, 32],
            "hybrid_32_64": [32, 64, 128, 64, 32],
            "hybrid_16_256_64": [16, 32, 64, 128, 64],
            # Adding support for the old filter organization options
          
        }

        # Get filter counts based on the selected filter organization
        filter_counts = filter_counts_map.get(filter_organization, [16, 32, 64, 128, 256])

        for filters in filter_counts:
            layers.append(nn.Conv2d(in_channels, filters, kernel_size=filter_size, padding=filter_size//2))
            if batchnorm:
                layers.append(nn.BatchNorm2d(filters))
            layers.append(activation_fn())
            layers.append(nn.MaxPool2d(2))
            in_channels = filters

        self.conv = nn.Sequential(*layers)
        self.flattened_size = filter_counts[-1] * (128 // (2**5))**2

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, dense_neurons),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(dense_neurons, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
