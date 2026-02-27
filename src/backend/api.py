from fastapi import APIRouter
from src.backend.schemas import PredictRequest, PredictResponse, Prediction
from src.backend.service import decode_image_b64, predict_image, MODEL_VERSION

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    predictions = []

    print("Len of images in request:", len(request.images))
    for image_payload in request.images:
        print("Processing image with size:", len(image_payload.image_b64))
        image_bytes = decode_image_b64(image_payload.image_b64)
        result = predict_image(image_bytes)
        
        print("result prediction:", result["probabilities"])

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
