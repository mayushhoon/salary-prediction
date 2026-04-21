import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model
model = joblib.load('best_model.pkl')

st.title('Salary Prediction App')
st.write('Predict your salary based on years of experience using the trained KNN model.')

# Input field
exp = st.number_input('Enter Experience Years:', min_value=0.0, max_value=50.0, value=1.0, step=0.1)

if st.button('Predict'):
    # Prepare input for prediction
    # We use a DataFrame to match the feature names used during training
    input_data = pd.DataFrame([[exp]], columns=['Experience Years'])
    prediction = model.predict(input_data)
    
    st.success(f'The predicted salary for {exp} years of experience is: ${prediction[0]:,.2f}')
