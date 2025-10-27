import pickle

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = f.read()
    model = pickle.loads(pipeline)

# The lead data to score
lead = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Make prediction
# predict_proba returns probabilities for both classes [prob_class_0, prob_class_1]
probability = model.predict_proba([lead])[0, 1]

print(f"Probability of conversion: {probability:.3f}")