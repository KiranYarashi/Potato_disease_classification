from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests  # Correct import for making external HTTP requests

# MODEL = tf.keras.models.load_model("saved_models/1.keras")
endpoint = "http://localhost:8501/v1/models/potato_model_1:predict"
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

app = FastAPI()

@app.get("/ping")
async def ping():
    return f"HI I'm alive and kicking"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))  # Convert the image into a numpy array [256,256,3]
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    # Prepare the payload for the Docker model
    payload = {
        "instances": image_batch.tolist()
    }

    # Send the request to the Docker model
    response = requests.post(endpoint, json=payload)
    
    # Print the response for debugging
   

    # Check if the response is successful
    if response.status_code != 200:
        return {"error": "Failed to get a valid response from the model server", "status_code": response.status_code}
    
    # Check if 'predictions' exists in the response
    if "predictions" not in response.json():
        return {"error": "Invalid response format. 'predictions' key not found."}

    predictions = response.json()["predictions"]

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
