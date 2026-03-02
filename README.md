# ml-model-integration-and-deployment
Containerized ML model with FastAPI and TorchScript export



# Run api with Dockerfile

To start Dockerfile you need to have docker downloaded on your computer. To use docker you need to open the docker application. You don't need to create an image because I pushed my docker image to dockerhub so you just need to run this code to download the image and start the api: "docker run -p 8000:8000 nore1/ml-api". Then you can go into the api using this link: "http://localhost:8000/docs". Now you can look around and try out our api and predict some base64 encoded pictures. 