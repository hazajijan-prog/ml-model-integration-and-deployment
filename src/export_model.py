import torch
from src.model import CNN
from pathlib import Path


def export_model():
    artifacts_path = Path("artifacts")
    weights_path = artifacts_path / "weights.pth"
    model_path = artifacts_path / "model.pt"

    if not weights_path.exists():
        raise RuntimeError("weights.pth not found. Train model first in K2.")

    # Skapa modellen
    model = CNN()

    # Ladda tränade vikter
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    # Eval mode
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 32, 32)

    # TorchScript
    scripted_model = torch.jit.trace(model, dummy_input)

    # Spara modellen
    artifacts_path.mkdir(exist_ok=True)
    scripted_model.save(model_path)

    print("Model exported successfully.")


if __name__ == "__main__":
    export_model()