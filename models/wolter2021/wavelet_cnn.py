"""
Wavelet Packet CNN architectures.

Contains both variants:
    - WaveletPacketCNN: Original architecture (512 -> 1, with dropout)
      Used by WolterWavelet2021Raw
    - WaveletPacketCNN128: Extended architecture (512 -> 128 -> 1, with embedding layer)
      Used by WolterWavelet2021_128

Preserved exactly from the original implementations.
"""
import torch.nn as nn
import torch.nn.functional as F


class WaveletPacketCNN(nn.Module):
    """
    Original CNN architecture from Wolter et al. 2022 paper.
    Simple CNN without ImageNet pretraining to fairly evaluate wavelet features.

    Classifier: Dropout(0.5) -> Linear(512, num_classes)
    Used by: WolterWaveletRawTrainer (WolterWavelet2021Raw)
    """

    def __init__(self, input_channels, num_classes=1):
        super(WaveletPacketCNN, self).__init__()

        # Convolutional layers - progressive architecture
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Original paper classifier: 512 -> 1 (direct classification)
        # No intermediate embedding layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x


class WaveletPacketCNN128(nn.Module):
    """
    CNN architecture for wavelet packet-based deepfake detection.
    Based on the architecture from Wolter et al. 2022.

    The paper uses a relatively simple CNN without ImageNet pretraining
    to fairly evaluate the wavelet packet features.

    Classifier: Linear(512, 128) -> ReLU -> Dropout(0.5) -> Linear(128, num_classes)
    Used by: WolterWavelet128Trainer (WolterWavelet2021_128)
    """

    def __init__(self, input_channels, num_classes=1):
        super(WaveletPacketCNN128, self).__init__()

        # Convolutional layers
        # The paper uses a progressive architecture
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head with dropout for regularization
        # Last two layers: 512 -> 128 (embeddings) -> 1 (binary classification)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x
