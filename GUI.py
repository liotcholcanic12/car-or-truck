!apt-get install xvfb
!xvfb-run -a python your_script.py
!pip install gradio
from tkinter import filedialog
from PIL import Image
import gradio as gr
import numpy as np
import tkinter as tk
import os


model = load_model('car_or_truck_vgg16.h5')

def preprocess_image(image):
  img = image.convert('RGB')
  img = img.resize((224, 224))
  img_array = np.array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  return img_array

def classify_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]

    if predicted_class == 0:
        return "Car"
    else:
        return "Truck"

with gr.Blocks() as iface:
    image_input = gr.Image(type="pil", label="Upload Image")
    output = gr.Textbox(label="Prediction")

    image_input.change(classify_image, inputs=image_input, outputs=output)

iface.launch()
