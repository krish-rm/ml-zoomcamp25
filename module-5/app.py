import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model
with open('pipeline_v1.bin', 'rb') as f:
    model = pickle.loads(f.read())

# Create FastAPI app
app = FastAPI()

# Define the input schema
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    # Convert to dict for prediction
    lead_dict = lead.dict()
    
    # Make prediction
    probability = model.predict_proba([lead_dict])[0, 1]
    
    return {
        "probability": float(probability),
        "subscription": bool(probability >= 0.5)
    }

@app.get("/")
def root():
    return {"message": "Lead Scoring API"}