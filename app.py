import streamlit as st
import joblib
import pandas as pd

# App title
st.title('💰 Salary Prediction App')

st.write('Predict your salary based on years of experience using a trained KNN model.')

# Load model safely
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_model.pkl')
        return model
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.error("❌ Model file not found. Please make sure 'best_model.pkl' is in the same folder.")
else:
    st.success("✅ Model loaded successfully!")

    # Input field
    exp = st.number_input(
        'Enter Years of Experience:',
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1
    )

    # Predict button
    if st.button('Predict Salary'):
        if exp <= 0:
            st.warning("⚠️ Please enter experience greater than 0.")
        else:
            try:
                # IMPORTANT: Match column name with training data
                input_data = pd.DataFrame([[exp]], columns=['Experience Years'])

                prediction = model.predict(input_data)

                st.success(f'💰 Predicted Salary: ${prediction[0]:,.2f}')

            except Exception as e:
                st.error("❌ Prediction failed. Check model input format or column names.")
