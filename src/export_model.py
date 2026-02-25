import torch
from src.model import SimpleClassifier
from pathlib import Path

def export_model():
    # 1. Skapa modellen
    model = SimpleClassifier()
    model.eval()

    # 2. Skapa dummy input (CIFAR-10 format)
    dummy_input = torch.randn(1, 3, 32, 32)

    # 3. Skapa TorchScript-modell
    scripted_model = torch.jit.trace(model, dummy_input)

    # 4. Säkerställ att artifacts-mappen finns
    artifacts_path = Path("artifacts")
    artifacts_path.mkdir(exist_ok=True)

    # 5. Spara modellen
    scripted_model.save(artifacts_path / "model.pt")

    print("Model exported successfully.")


if __name__ == "__main__":
    export_model()