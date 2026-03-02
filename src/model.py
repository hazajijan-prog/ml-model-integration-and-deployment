"""
Model module.

Defines neural network architectures used for CIFAR-10 classification.
Includes both a simple feedforward baseline and the CNN used in production.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    """
    Simple feedforward baseline model for CIFAR-10.

    Architecture:
        - Input: 3x32x32 images (flattened to 3072 features)
        - Hidden layer: 128 neurons + ReLU
        - Output: 10 classes
    """


    def __init__(self):
        super().__init__()

        # Fully connected network for baseline comparison
        self.model = nn.Sequential(
            nn.Linear(3072, 128), 
            nn.ReLU(), 
            nn.Linear(128, 10), 
        )

    def forward(self, x):
        """
        Defines forward pass of the baseline model.
        """

        # Flatten image tensor before passing to linear layers
        x = x.view(x.size(0), -1)

        return self.model(x)

class CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification.

    Architecture:
        - 2 convolutional layers with ReLU + MaxPooling
        - 2 fully connected layers
        - Output: raw logits for 10 classes
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Downsampling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected classifier
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass of the CNN.

        Returns raw logits (softmax is applied during inference).
        """
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten feature maps before fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x