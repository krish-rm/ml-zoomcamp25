from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import onnxruntime as ort
import math


# ---------- Image download ----------

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


# ---------- Preprocessing ----------
# target size from Q2 = (200, 200)

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


# Convert to numpy & normalize like training
def preprocess(img):
    img = np.array(img).astype(np.float32) / 255.0   # scale
    img = np.transpose(img, (2, 0, 1))              # HWC → CHW
    img = np.expand_dims(img, 0)                    # add batch dim
    return img


# ---------- Load Model ----------
session = ort.InferenceSession("hair_classifier_v1.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("Input:", input_name)
print("Output:", output_name)


# ---------- Run prediction ----------
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(url)
img = prepare_image(img)
x = preprocess(img)

raw = session.run([output_name], {input_name: x})[0][0][0]
prob = 1 / (1 + math.exp(-raw))   # sigmoid

print("\nRaw model output:", raw)
print("Probability (sigmoid):", prob)

if prob > 0.5:
    print("→ Predicted: Curly hair")
else:
    print("→ Predicted: Straight hair")
