from fastapi import FastAPI 
# from src.utils.constants import MODEL_PATH
from schemas import PredictRequest, PredictResponse
from service import decode_image_b64, predict_stub, MODEL_VERSION

app = FastAPI(title="CIFAR10 API")

images_data = []

@app.get("/images/")
async def load_images():
    return images_data

@app.post("/images/predict", response_model=PredictResponse)
def predict_images(req: PredictRequest) -> PredictResponse:
    # 1) "Packa upp" alla bilder: base64 -> bytes
    decoded = [decode_image_b64(img.image_b64) for img in req.images]

    # (just nu gör vi inget med decoded-bytes, men senare skickas de till modellen)
    _ = decoded

    # 2) Få predictions (stub nu)
    preds = predict_stub(len(req.images))

    # 3) Skicka tillbaka svar i exakt format
    return PredictResponse(predictions=preds, model_version=MODEL_VERSION)