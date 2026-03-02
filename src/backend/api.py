"""
API layer for model inference.

Defines HTTP endpoints and maps incoming requests
to the service layer for prediction.
"""
from fastapi import APIRouter
from src.backend.schemas import PredictRequest, PredictResponse, Prediction
from src.backend.service import decode_image_b64, predict_image, MODEL_VERSION

# Router instance for grouping prediction-related endpoints
router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    
    """
    Accepts one or more base64-encoded images and returns
    predicted label and class probabilities.

    Flow:
    - Decode base64 image
    - Run inference via service layer
    - Map result to response schema
    """

    predictions = []

    for image_payload in request.images:
        # Decode incoming base64 image into raw bytes
        image_bytes = decode_image_b64(image_payload.image_b64)
        # Run inference using the service layer
        result = predict_image(image_bytes)

        # Map internal result to API response schema
        predictions.append(
            Prediction(
                label=result["label"],
                probabilities=result["probabilities"]
    )
)

    return PredictResponse(
        predictions=predictions,
        model_version=MODEL_VERSION
    )
