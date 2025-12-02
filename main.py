import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # ALLOW_ORIGINS: Ubah jadi bintang ["*"] agar semua IP boleh masuk
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
MODEL_PATH = "model_effnet_cpu.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

target_size = (224, 224)
CLASS_NAMES = ["apple", "banana", "cabbage", "carrot", "grapes", "kiwi", "lettuce", "mango", "onion", "orange", "pear", "potato", "spinach", "tomato"]

def read_imagefile(file_data) -> np.ndarray:
    image = Image.open(BytesIO(file_data)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0) 
    return image_array 

@app.get("/")
def home():
    return {"message" : "Server Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    processed_image = read_imagefile(image_data)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    result_class = CLASS_NAMES[predicted_class_index]

    confidence = float(np.max(prediction[0]))

    return {
        "predicted_class": result_class,
        "confidence": confidence}