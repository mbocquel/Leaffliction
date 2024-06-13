from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from model.my_CNN_model import My_CNN_model
from configs.CFG import CFG


app = FastAPI()

MODEL = My_CNN_model(CFG)
MODEL.class_names = [
    "Apple_Black_rot",
    "Apple_healthy",
    "Apple_rust",
    "Apple_scab",
    "Grape_Black_rot",
    "Grape_Esca",
    "Grape_healthy",
    "Grape_spot",
]
MODEL.load("saved_models/my_cnn_model.keras")


@app.get("/ping")
async def ping():
    return "Hello I am alive"


def read_file_as_image(data) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    prediction, confidence = MODEL.predict_one_img(image, print=False)
    print(prediction)
    if confidence:
        confidence = round(confidence, 3)
    return {"prediction": prediction, "confidence": confidence}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8888)
