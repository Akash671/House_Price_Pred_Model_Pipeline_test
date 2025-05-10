import gradio as gr
import pickle # Changed from joblib
import numpy as np
import os

# Define paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, 'house_price_model.pkl') # Changed extension
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl') # Changed extension

# Load the trained model and scaler
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f) # Use pickle.load
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f) # Use pickle.load

def predict_house_price(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    features_unscaled = np.array([med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]).reshape(1, -1)
    features_scaled = scaler.transform(features_unscaled)
    prediction = model.predict(features_scaled)
    return prediction[0]

# Define the Gradio interface
inputs = [
    gr.Slider(minimum=0, maximum=10, step=0.1, label="MedInc (Median income in block group)"),
    gr.Slider(minimum=1, maximum=55, step=1, label="HouseAge (Median house age in block group)"),
    gr.Slider(minimum=1, maximum=150, step=0.1, label="AveRooms (Average number of rooms per household)"),
    gr.Slider(minimum=0.5, maximum=35, step=0.1, label="AveBedrms (Average number of bedrooms per household)"),
    gr.Slider(minimum=3, maximum=40000, step=10, label="Population (Block group population)"),
    gr.Slider(minimum=0.5, maximum=60, step=0.1, label="AveOccup (Average number of household members)"),
    gr.Slider(minimum=32, maximum=42, step=0.0001, label="Latitude"),
    gr.Slider(minimum=-125, maximum=-114, step=0.0001, label="Longitude"),
]

output = gr.Label(label="Predicted House Price (in $100,000s)")

description = "Predict house prices based on the California Housing dataset features. Adjust the sliders and see the predicted price."
article = "<p style='text-align: center;'>Model trained on scaled data. Input values are scaled before prediction.</p>"

gr.Interface(
    fn=predict_house_price,
    inputs=inputs,
    outputs=output,
    title="California House Price Prediction",
    description=description,
    article=article
).launch()