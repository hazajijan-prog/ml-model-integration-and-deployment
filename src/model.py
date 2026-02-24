"""
Model-modul.

Innehåller en fullt kopplad modell för klassificering av CIFAR-10.
"""

import torch.nn as nn


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
