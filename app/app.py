from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load the trained model
model = load_model('smoking_detector.h5')

def prepare_image(image: Image.Image):
    # Convert image to RGB if it has an alpha channel
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def main():
    html_content = """
    <html>
    <head>
        <title>Smoking Detection</title>
        <style>
            body {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
            }
            h2 {
                margin-bottom: 20px;
            }
            form {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            input[type="file"] {
                margin-bottom: 20px;
            }
            input[type="submit"] {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 5px;
                transition: background-color 0.3s;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
            #result {
                margin-top: 20px;
                text-align: center;
            }
            #uploaded-image {
                max-width: 300px;
                margin-top: 20px;
            }
            .corner-image {
                position: absolute;
                top: 10px;
                width: 200px; /* Adjust size as needed */
            }
            #left-image {
                left: 10px;
            }
            #right-image {
                right: 10px;
            }
        </style>
        <script>
            function previewImage(event) {
                const image = document.getElementById('uploaded-image');
                image.src = URL.createObjectURL(event.target.files[0]);
                image.style.display = 'block';
            }
            async function submitForm(event) {
                event.preventDefault(); // Prevent page refresh
                const formData = new FormData(event.target);
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });
                const result = await response.text();
                document.getElementById('result').innerHTML = result;
            }
        </script>
    </head>
    <body>
        <img id="left-image" class="corner-image" src="Z:\Semster 6.5 A+\Adv. AI\smoking.jpg">   
        <img id="right-image" class="corner-image" src="">
        <form onsubmit="submitForm(event)">
            <input name="file" type="file" accept="image/*" onchange="previewImage(event)" required>
            <input type="submit" value="Detect Smoking">
        </form>
        <img id="uploaded-image" src="" alt="Uploaded Image" style="display:none;">
        <div id="result"></div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    processed_image = prepare_image(image)
    prediction = model.predict(processed_image)
    result = "Smoking" if np.argmax(prediction) == 1 else "Not Smoking"
    return HTMLResponse(content=f"<h3>Prediction: {result}</h3>")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
