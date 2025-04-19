import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, filter_size=3, filter_organization='16_256', num_classes=10,   
                 dense_neurons=256, activation_fn=nn.Mish, dropout=0.0, batchnorm=True):
        super(CNN, self).__init__()
        layers = []
        in_channels = 3

        # Use predefined filter configurations
        if filter_organization == "16_256":
            filter_counts = [16, 32, 64, 128, 256]
        elif filter_organization == "32_512":
            filter_counts = [32, 64, 128, 256, 512]
        elif filter_organization == "constant_32":
            filter_counts = [32] * 5
        elif filter_organization == "pyramid_256_16":
            filter_counts = [256, 128, 64, 32, 16]
        elif filter_organization == "64_1024":
            filter_counts = [64, 128, 256, 512, 1024]
        elif filter_organization == "pyramid_128_8":
            filter_counts = [128, 64, 32, 16, 8]
        elif filter_organization == "pyramid_512_32":
            filter_counts = [512, 256, 128, 64, 32]
        elif filter_organization == "hybrid_32_64":
            filter_counts = [32, 64, 128, 64, 32]
        elif filter_organization == "hybrid_16_256_64":
            filter_counts = [16, 32, 64, 128, 64]
        else:
            filter_counts = [16, 32, 64, 128, 256]  # default

        # Convolutional layers
        for out_channels in filter_counts:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=filter_size // 2))
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_fn())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Flatten size: image shrinks by 2^5 due to 5 maxpool layers
        final_feat_dim = 128 // (2 ** 5)
        self.flattened_size = filter_counts[-1] * final_feat_dim * final_feat_dim

        # Fully connected layers
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
