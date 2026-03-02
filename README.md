# M3 – Model Integration and Deployment

## Overview

This project demonstrates a professional MLOps workflow where a trained deep learning model is integrated into a REST API and deployed via Docker containers.

The selected model is a Convolutional Neural Network (CNN) trained on CIFAR-10, chosen from the models developed in K2.

The pipeline includes:
- PyTorch model (trained in K2)
- TorchScript export for deployment
- FastAPI inference service
- Docker containerization using `uv`
- Feature branches and code reviews

---

## Model

The model used is a CNN trained on CIFAR-10 (32x32 RGB images).

- Input: 3x32x32 images
- Output: 10 class logits
- Accuracy (K2 test set): ~60–70%

The model is exported to TorchScript using:
```bash
python -m src.export_model
```

This generates `artifacts/model.pt`. The API loads this TorchScript model during startup.

---

## API

The FastAPI service exposes:

### POST `/predict`

Accepts JSON with base64-encoded images:
```json
{
  "images": [
    {
      "image_b64": "BASE64_ENCODED_IMAGE"
    }
  ]
}
```

Returns:
```json
{
  "predictions": [
    {
      "label": "dog",
      "probabilities": [0.01, 0.02, "..."]
    }
  ],
  "model_version": "cnn-cifar10-v1.0"
}
```

Softmax is applied during inference to convert logits to probabilities.

---

## Running Locally

### 1. Export model
```bash
python -m src.export_model
```

### 2. Start API
```bash
uv run uvicorn main:app --reload
```

Access Swagger UI at http://localhost:8000/docs

---

## Docker

To get started, make sure Docker is installed and the **Docker application is running** before executing any commands.

### Build image
```bash
docker build -t ml-api .
```

This installs everything from the Dockerfile and creates an image called `ml-api`.

### Run container
```bash
docker run -p 8000:8000 ml-api
```

### Access the API

Go to `http://localhost:8000/docs` to explore and test the API. You can try out the `/predict` endpoint by sending base64-encoded images.

---

## Project Structure
```
src/
├── model.py             # Model architecture
├── export_model.py      # TorchScript export utility
├── backend/
│   ├── api.py           # FastAPI endpoints
│   ├── service.py       # Inference logic
│   └── schemas.py       # Request/response models
artifacts/
├── weights.pth          # Trained weights (from K2)
└── model.pt             # TorchScript model
```

---

## Git Workflow & Code Review

Development was done using feature branches and pull requests.

Key pull requests:

- **[PR #8 – Feature/load trained weights](https://github.com/hazajijan-prog/ml-model-integration-and-deployment/pull/8)**  
  Integrated trained CNN model and implemented TorchScript inference pipeline.

- **[PR #7 – docker-file](https://github.com/hazajijan-prog/ml-model-integration-and-deployment/pull/7)**  
  Added Dockerfile and containerized the FastAPI service.

- **[PR #5 – Feature/api structure](https://github.com/hazajijan-prog/ml-model-integration-and-deployment/pull/5)**  
  Designed API structure and implemented prediction endpoint.

- **[PR #10 – Add documentation and explanatory comments](https://github.com/hazajijan-prog/ml-model-integration-and-deployment/pull/10)**  
  Improved readability and maintainability with detailed documentation.

---

## Checklist

- Container builds and runs
- API returns correct predictions via POST /predict
- TorchScript model export implemented
- Feature branches and code reviews used
- README contains documented PR links

---

## Authors

- Haza Jijan 
- Nore Lindkvist