import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('employee_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app layout
st.set_page_config(page_title="Employee Churn Prediction", layout="centered")

# Title of the app
st.markdown("<h1 style='color: #4CAF50;'>Employee Churn Prediction App</h1>", unsafe_allow_html=True)

# Sidebar inputs for employee details
st.sidebar.markdown("### Enter Employee Details")

# Input fields for user features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=18.0, max_value=120.0, value=50.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=8684.0, value=500.0)


# Preprocessing function for user input
def preprocess_user_input(gender, senior_citizen, tenure, contract, payment_method, monthly_charges, total_charges):
    # Create a dictionary with the user input
    user_data = {
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'tenure': [tenure],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    }

    # Create a DataFrame from the input
    user_input_df = pd.DataFrame(user_data)
    
    # One-hot encode the input data
    user_input_df = pd.get_dummies(user_input_df)
    
    # Align the user input with the model's feature columns (handle missing columns)
    expected_columns = pd.read_csv('model_feature_columns.csv')
    
    # Add missing columns with zero values
    user_input_df = user_input_df.reindex(columns=expected_columns, fill_value=0)
    
    # Scale the input using the same scaler
    user_input_scaled = scaler.transform(user_input_df)
    
    return user_input_scaled

# Prediction function
if st.sidebar.button("Predict"):
    # Preprocess user input
    user_input = preprocess_user_input(gender, senior_citizen, tenure, contract, payment_method, monthly_charges, total_charges)
    
    # Make prediction
    prediction = model.predict(user_input)[0]
    
    # Display prediction result
    if prediction==1:
        st.markdown("<h3 style='color: #F44336;'>The employee is likely to leave the company.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: #4CAF50;'>The employee is likely to stay in the company.</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h4 style='color: #2596F3;'>Please enter employee details in the sidebar and press 'Predict'.</h4>", unsafe_allow_html=True)

