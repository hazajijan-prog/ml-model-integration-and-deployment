from fastapi import APIRouter
from src.backend.schemas import PredictRequest, PredictResponse, Prediction
from src.backend.service import decode_image_b64, predict_image, MODEL_VERSION

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    predictions = []

    for image_payload in request.images:
        image_bytes = decode_image_b64(image_payload.image_b64)
        result = predict_image(image_bytes)

        predictions.append(
            Prediction(
                label=result["label"]
            )
        )

    return PredictResponse(
        predictions=predictions,
        model_version=MODEL_VERSION
    )