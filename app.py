import gradio as gr
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model/house_price_model.joblib')

def predict_house_price(med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude):
    features = np.array([med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]).reshape(1, -1)
    # Important: The model was trained on scaled data.
    # You need to apply the same scaling to the input features here.
    # This requires saving the scaler from your pipeline.py and loading it here.
    # For now, this will likely produce inaccurate predictions without scaling.
    prediction = model.predict(features)
    return prediction[0]

# Define the Gradio interface
inputs = [
    gr.Slider(minimum=0, maximum=10, label="MedInc"),
    gr.Slider(minimum=0, maximum=10, label="HouseAge"),
    gr.Slider(minimum=0, maximum=10, label="AveRooms"),
    gr.Slider(minimum=0, maximum=10, label="AveBedrms"),
    gr.Slider(minimum=0, maximum=100000, label="Population"),
    gr.Slider(minimum=0, maximum=10, label="AveOccup"),
    gr.Slider(minimum=-90, maximum=90, label="Latitude"),
    gr.Slider(minimum=-180, maximum=180, label="Longitude"),
]

output = gr.Label(label="Predicted House Price")

gr.Interface(fn=predict_house_price, inputs=inputs, outputs=output, title="House Price Prediction").launch()