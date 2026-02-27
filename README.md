# ml-model-integration-and-deployment
Containerized ML model with FastAPI and TorchScript export



to start Dockerfile you need to have docker downloaded on your computer. After that you need to create an image: "docker build -t ml-api ." this will install everything from Dockerfile and create an image called ml-api. After that you can start the api using: "docker run -p 8000:8000 ml-api". Then you can go into the api using this link: "http://localhost:8000/docs". Now you can look around and try out our api and predict some base64 encoded pictures. 