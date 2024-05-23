from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from model.my_CNN_model import My_CNN_model
from configs.CFG import CFG



app = FastAPI()

MODEL = My_CNN_model(CFG)
MODEL.load_data()
MODEL.load("saved_models/my_cnn_model.keras")


@app.get("/ping")
async def ping():
	return "Hello I am alive"

def read_file_as_image(data) -> np.ndarray:
	return np.array(Image.open(BytesIO(data)))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
	image = read_file_as_image(await file.read())
	prediction = MODEL.predict_one_img(image,print=False)
	return prediction


if __name__ == "__main__":
	uvicorn.run(app, host='localhost', port=4242)