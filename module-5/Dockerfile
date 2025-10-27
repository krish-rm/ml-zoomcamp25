FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

# Copy your model and FastAPI app
COPY pipeline_v1.bin .
COPY app.py .

# Install dependencies
RUN pip install fastapi uvicorn requests scikit-learn

EXPOSE 9696

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]
