"""
Model-modul.

Innehåller en fullt kopplad modell för klassificering av CIFAR-10.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleClassifier(nn.Module):
    """
    Enkel feedforward-modell för bildklassificering.

    Arkitektur:
        - Input: 3x32x32 bilder (flattenas till 3072 värden)
        - Doldt lager: 128 neuroner + ReLU
        - Output: 10 klasser
    """

    def __init__(self):
        super().__init__()

        # Definierar nätverkets lager i ordning
        self.model = nn.Sequential(
            nn.Linear(3072, 128),  # 3*32*32 -> 128 neuroner
            nn.ReLU(),  # Aktiveringsfunktion
            nn.Linear(128, 10),  # 10 utgångar (en per klass)
        )

    def forward(self, x):
        """
        Definierar hur data flödar genom nätverket.
        """

        # Plattar ut bilden till en vektor innan den skickas in i nätverket
        x = x.view(x.size(0), -1)

        return self.model(x)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x