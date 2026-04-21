import streamlit as st
import joblib
import pandas as pd

# Title
st.title("💰 Salary Prediction App")

st.write("Predict your salary based on years of experience.")

# Load model safely
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_model.pkl")
    except:
        return None

model = load_model()

# Check model
if model is None:
    st.error("❌ Model not found. Upload 'best_model.pkl'")
else:
    st.success("✅ Model loaded successfully")

    # Input
    exp = st.number_input(
        "Enter Years of Experience:",
        min_value=0.0,
        max_value=50.0,
        value=1.0,
        step=0.1
    )

    # Predict button
    if st.button("Predict Salary"):

        if exp <= 0:
            st.warning("⚠️ Enter experience greater than 0")
        else:
            try:
                # IMPORTANT: match column name with training
                input_data = pd.DataFrame([[exp]], columns=['Experience Years'])

                prediction = model.predict(input_data)

                # USD salary
                usd_salary = prediction[0]

                # Convert to INR
                usd_to_inr = 83  # you can update this anytime
                inr_salary = usd_salary * usd_to_inr

                # Output
                st.success(f"""
💰 Predicted Salary:
- USD: ${usd_salary:,.2f}
- INR: ₹{inr_salary:,.2f}
""")

            except Exception as e:
                st.error("❌ Prediction failed. Check column name or model.")
