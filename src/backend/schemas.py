from pydantic import BaseModel, Field
from typing import List, Optional


class ImagePayload(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image (PNG/JPG).")


class PredictRequest(BaseModel):
    images: List[ImagePayload] = Field(..., min_length=1)


class Prediction(BaseModel):
    label: str
    probabilities: Optional[List[float]] = None


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    model_version: str