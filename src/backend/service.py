from typing import List
import base64
import re
from fastapi import HTTPException
from src.backend.schemas import Prediction
from src.utils.constants import MODEL_PATH
import torch
from PIL import Image
import io
from torchvision import transforms


CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_VERSION = "cnn-cifar10-v1.0"

if not MODEL_PATH.exists():
    raise RuntimeError("Model file not found. Run export_model first.")

model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

def predict_image(image_bytes:bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invaid image format")
    
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = probs.argmax(dim=1).item()
        probabilities = [round(float(p),4) for p in probs.squeeze()]

    return {
    "label": CIFAR10_LABELS[predicted_class],
    "class_index": predicted_class,
    "probabilities": probabilities,
    "model_version": MODEL_VERSION
}

DATA_URL_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,")


def decode_image_b64(image_b64: str) -> bytes:
    cleaned = DATA_URL_RE.sub("", image_b64.strip())
    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload")


def predict_stub(num_images: int) -> List[Prediction]:
    pass