import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

# Load trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "car_price_model.pkl")
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🚗 Car Price Prediction Web App")
st.write("Enter the specifications of a car below to predict its estimated market price using our Machine Learning model.")

st.markdown("---")

# User Input Fields
st.subheader("Car Specifications")

col1, col2 = st.columns(2)
with col1:
    enginesize = st.number_input("Engine Size", min_value=50, max_value=500, value=120, step=10, help="Volume of all the cylinders in the engine (in cubic inches)")
    horsepower = st.number_input("Horsepower", min_value=40, max_value=500, value=100, step=5, help="Engine's power output")
    carwidth = st.number_input("Car Width (inches)", min_value=60.0, max_value=80.0, value=65.5, step=0.5)

with col2:
    curbweight = st.number_input("Curb Weight (lbs)", min_value=1000, max_value=4500, value=2500, step=100)
    citympg = st.number_input("City MPG", min_value=10, max_value=60, value=25, step=1)
    highwaympg = st.number_input("Highway MPG", min_value=10, max_value=70, value=30, step=1)

# Convert inputs to DataFrame
input_data = pd.DataFrame(
    [[enginesize, horsepower, carwidth, curbweight, citympg, highwaympg]],
    columns=["enginesize", "horsepower", "carwidth", "curbweight", "citympg", "highwaympg"]
)

st.markdown("---")

if st.button("Predict Price", type="primary", use_container_width=True):
    with st.spinner("Processing prediction..."):
        # Make Prediction
        prediction = model.predict(input_data)
        
    st.success(f"### Predicted Car Price: ${prediction[0]:,.2f}")
    st.balloons()
    
st.sidebar.header("About")
st.sidebar.info(
    "This web app is part of a machine learning workflow that predicts "
    "used car prices based on their technical specifications.\\n\\n"
    "Model used: **Random Forest Regressor** \\n"
    "(Trained with an R² accuracy of > 95% on the test set)."
)
