import streamlit as st
import joblib
import pandas as pd

st.title("💰 Salary Prediction App")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# Input
exp = st.number_input("Enter Years of Experience:", 0.0, 50.0, 1.0)

# Predict
if st.button("Predict"):
    input_data = pd.DataFrame([[exp]], columns=['Experience Years'])
    prediction = model.predict(input_data)

    st.success(f"💰 Predicted Salary: ${prediction[0]:,.2f}")
