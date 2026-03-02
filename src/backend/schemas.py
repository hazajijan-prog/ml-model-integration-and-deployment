"""
Pydantic schemas for request and response validation.

Defines the structure of incoming prediction requests
and outgoing prediction responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class ImagePayload(BaseModel):
    """
    Represents a single base64-encoded image input.
    """
    image_b64: str = Field(..., description="Base64-encoded image (PNG/JPG).")


class PredictRequest(BaseModel):
    """
    Request model containing one or more images for prediction.
    """
    images: List[ImagePayload] = Field(..., min_length=1)


class Prediction(BaseModel):
    """
    Prediction result for a single image.
    """
    label: str
    probabilities: Optional[List[float]] = None


class PredictResponse(BaseModel):
    """
    Response model returned by the /predict endpoint.
    """
    predictions: List[Prediction]
    model_version: str