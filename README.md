# SMOKE-DETECTION: END TO END ML PROJECT
![Docker Image](smoke.gif)
## Overview
- Data Preparation: Images from Kaggle are organized into "smoking" and "notsmoking" folders.
- Model Training: A VGG16 model is used for image classification, fine-tuned with your dataset.
- Deployment: The model is integrated into a FastAPI application.
- Containerization: The application and its dependencies are containerized using Docker.
- Exposing the App: Ngrok is used to create a secure tunnel to access the local FastAPI app from anywhere.
- Testing: Users can upload images through a web interface to get real-time predictions.
