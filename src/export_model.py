"""
Model export utility.

Converts a trained PyTorch CNN model into TorchScript format
for deployment in the API service.
"""

import torch
from src.model import CNN
from pathlib import Path


def export_model():
    """
    Loads trained weights and exports the model to TorchScript format.

    Steps:
    - Load trained CNN weights
    - Switch to evaluation mode
    - Trace model using dummy input
    - Save TorchScript model to artifacts/
    """
    artifacts_path = Path("artifacts")
    weights_path = artifacts_path / "weights.pth"
    model_path = artifacts_path / "model.pt"

    # Ensure trained weights exist before export
    if not weights_path.exists():
        raise RuntimeError("weights.pth not found. Train model first in K2.")

    # Initialize model architecture
    model = CNN()

    # Load trained weights from training pipeline (K2)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    # Set model to evaluation mode (disables dropout/batchnorm training behavior)
    model.eval()

    # Dummy input required for TorchScript tracing
    # Shape must match training input dimensions (CIFAR-10: 3x32x32)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Convert model to TorchScript for production deployment
    scripted_model = torch.jit.trace(model, dummy_input)

    # Ensure artifacts directory exists and save exported model
    artifacts_path.mkdir(exist_ok=True)
    scripted_model.save(model_path)

    print("Model exported successfully.")


if __name__ == "__main__":
    export_model()