from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import time

app = FastAPI()

origins = [
    "https://fruit-quality-detection-frontend-am.vercel.app",  
    "http://localhost:5173",  
]

# Allow React frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load multi-output model
model = tf.keras.models.load_model("model/fruit_multi_output_model_1.h5")

# Fruit class mapping (order must match training)
fruit_labels = ["apple", "banana", "orange"]

# Prewarm the model
model.predict(np.zeros((1, 100, 100, 3)))

# Preprocess image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((100, 100))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    contents = await file.read()
    image = preprocess_image(contents)

    # Predict freshness and fruit type
    freshness_pred, fruit_pred = model.predict(image)
    
    # Extract predictions
    freshness_score = float(freshness_pred[0][0])
    predicted_fruit_index = np.argmax(fruit_pred[0])
    predicted_fruit_name = fruit_labels[predicted_fruit_index]

    # Convert to % format
    fresh_percent = round(freshness_score * 100, 2)
    rotten_percent = round(100 - fresh_percent, 2)

    print("Prediction done in:", round(time.time() - start_time, 3), "seconds")
    print("Freshness:", fresh_percent, "% | Fruit:", predicted_fruit_name)

    return JSONResponse(content={
        "freshness": fresh_percent,
        "rotten": rotten_percent,
        "fruit": predicted_fruit_name
    })

