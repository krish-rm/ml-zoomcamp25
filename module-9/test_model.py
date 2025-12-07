from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import onnxruntime as ort

# === IMAGE DOWNLOAD & PROCESSING ===

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):  # <-- Fix here
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def to_numpy(img):
    arr = np.array(img)/255.0
    arr = np.transpose(arr,(2,0,1))  # HWC→CHW
    arr = np.expand_dims(arr,0).astype(np.float32)
    return arr

# === LOAD MODEL ===
session = ort.InferenceSession("hair_classifier_v1.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("INPUT NAME:", input_name)
print("OUTPUT NAME:", output_name)

# === TEST IMAGE ===
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(url)
img = prepare_image(img)
x = to_numpy(img)

# === RUN MODEL ===
pred = session.run([output_name], {input_name: x})
print("\nPrediction:", pred[0])
