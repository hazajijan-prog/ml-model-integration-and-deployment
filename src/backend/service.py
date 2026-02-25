from typing import List
import base64
import re
from fastapi import HTTPException
from schemas import Prediction
from utils.constants import MODEL_PATH

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_VERSION = "stub-0.1"

DATA_URL_RE = re.compile(r"^data:image\/[a-zA-Z0-9.+-]+;base64,")


def decode_image_b64(image_b64: str) -> bytes:
    cleaned = DATA_URL_RE.sub("", image_b64.strip())
    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload")


def predict_stub(num_images: int) -> List[Prediction]:
    pass