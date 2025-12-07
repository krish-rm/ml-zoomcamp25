import os
import json
import base64
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image
import onnxruntime as ort
import math


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess(img):
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr


# Lambda entry point
def lambda_handler(event, context=None):

    url = event["url"]
    img = download_image(url)
    img = prepare_image(img)
    x = preprocess(img)

    session = ort.InferenceSession("hair_classifier_empty.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: x})[0][0][0]

    return {
        "prediction_raw": float(result),
        "message": "Hair type score returned"
    }
