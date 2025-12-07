FROM agrigorev/model-2025-hairstyle:v1

# Install dependencies
RUN pip install pillow onnxruntime

COPY lambda_handler.py .

CMD ["lambda_handler.lambda_handler"]
