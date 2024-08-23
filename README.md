# SMOKE-DETECTION: END TO END ML PROJECT
<div align="center">
  <img src="smoke.gif" alt="image" width="500" />
</div>

## Overview
- Data Preparation: Images from Kaggle are organized into "smoking" and "notsmoking" folders.
- Model Training: A VGG16 model is used for image classification, fine-tuned with your dataset.
- Deployment: The model is integrated into a FastAPI application.
- Containerization: The application and its dependencies are containerized using Docker.
- Exposing the App: Ngrok is used to create a secure tunnel to access the local FastAPI app from anywhere.
- Testing: Users can upload images through a web interface to get real-time predictions.

## Data Preparation
### 1. Get Data: KaggleAPI
```bash
#make sure to change username and dataset name
kaggle datasets download -d vitaminc/cigarette-smoker-detection
```
```bash
unzip cigarette-smoker-detection.zip -d dataset
```
###  2.Organize the Dataset:
Ensure your dataset has the following structure:
```bash
dataset/
├── smoking/
└── notsmoking/
```
### 3.Preprocess the Data:
- Resize & Normalize:
  ```python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator

  datagen = ImageDataGenerator(rescale=1./255)
  train_generator = datagen.flow_from_directory(
      'dataset/',
      target_size=(224, 224),
      batch_size=32,
      class_mode='binary'
  )
  ```
-  Split Data:
    ```python
    from sklearn.model_selection import train_test_split
    image_paths, labels = # Your image paths and labels
    train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42
    )
    ```
### 4.Data Augmentation:
```python
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```
### 5.Create Data Loaders:
```python
train_generator = datagen.flow_from_directory(
    'dataset/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```


   

