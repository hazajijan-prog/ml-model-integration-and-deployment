"""
Service layer for model inference.

Responsible for:
- Loading the exported TorchScript model
- Preprocessing incoming images
- Running inference
- Returning structured prediction results
"""

import base64
import re
from fastapi import HTTPException
from src.utils.constants import MODEL_PATH
import torch
from PIL import Image
import io
from torchvision import transforms

# Order must match the original CIFAR-10 class index mapping.
# The model outputs logits in this exact index order.
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_VERSION = "cnn-cifar10-v1.0"

# Ensure exported TorchScript model exists before loading.
# The model is created via export_model.py during deployment preparation.
if not MODEL_PATH.exists():
    raise RuntimeError("Model file not found. Run export_model first.")

# Load TorchScript model for production inference (CPU execution).
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# Resize to 32x32 to match CIFAR-10 training resolution
# and convert image to tensor for model input.
preprocess = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

def predict_image(image_bytes:bytes):
    """
    Perform inference on a single image.

    Steps:
    - Decode image bytes
    - Apply preprocessing
    - Run forward pass through TorchScript model
    - Apply softmax to obtain class probabilities

    Returns:
        dict containing:
            - label (str)
            - class_index (int)
            - probabilities (List[float])
            - model_version (str)
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invaid image format")
    
    tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        
        # Convert logits to probabilities using softmax
        probs = torch.softmax(outputs, dim=1)
        
        predicted_class = probs.argmax(dim=1).item()
        # Round probabilities for cleaner API output
        probabilities = [round(float(p),4) for p in probs.squeeze()]

    return {
    "label": CIFAR10_LABELS[predicted_class],
    "class_index": predicted_class,
    "probabilities": probabilities,
    "model_version": MODEL_VERSION
}

DATA_URL_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,")


def decode_image_b64(image_b64: str) -> bytes:
    """
    Decode a base64-encoded image string into raw bytes.

    Removes potential data URL prefix before decoding.
    Raises HTTPException if payload is invalid.
    """
    
    cleaned = DATA_URL_RE.sub("", image_b64.strip())
    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload")