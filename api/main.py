from fastapi import FastAPI , File , UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

MODEL = tf.keras.models.load_model("saved_models/aa/1.keras")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

app = FastAPI()

@app.get("/ping")
async def ping():
    return f"HI im alive and kicking"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))               #convert the image into numpy array[256,256,3]
    return image            
    

@app.post("/prediction")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)                    #make the dimensions [1,2]  --> [[1,2]]
    predictions = MODEL.predict(image_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  #get the class name  of prediction[0.23464, 0.934, 0.123]
    confidence = np.max(predictions[0])                       #get the max value among the prediction
    
    
    return  {
        "class": predicted_class,
        "confidence": float(confidence)
    }
    
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)