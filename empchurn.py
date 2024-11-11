import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Employee Churn Prediction",
    page_icon="🏢",  # Use the emoji directly
    layout="centered"  # Optional; defaults to centered
)

# Load saved logistic regression model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler_churn.joblib')

# Define feature columns used in training (ensure the same order)
feature_columns = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 
    'MonthlyCharges', 'TotalCharges', 'InternetService_DSL', 'InternetService_Fiber optic',
    'InternetService_No', 'Contract_Month-to-month', 'Contract_One year', 
    'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]

# Streamlit app title
st.title('Employee Churn Prediction')

# Input fields for user to enter data
gender = st.selectbox('Gender', ['Female', 'Male'])
SeniorCitizen = st.selectbox('Senior Citizen (Yes = 1 & No = 0)', [0, 1])
Partner = st.selectbox('Partner', ['No', 'Yes'])
Dependents = st.selectbox('Dependents', ['No', 'Yes'])
tenure = st.slider('Tenure', 0, 72, 1)
PhoneService = st.selectbox('Phone Service', ['No', 'Yes'])
MultipleLines = st.selectbox('Multiple Lines', ['No', 'Yes'])
OnlineSecurity = st.selectbox('Online Security', ['No', 'Yes'])
OnlineBackup = st.selectbox('Online Backup', ['No', 'Yes'])
DeviceProtection = st.selectbox('Device Protection', ['No', 'Yes'])
TechSupport = st.selectbox('Tech Support', ['No', 'Yes'])
StreamingTV = st.selectbox('Streaming TV', ['No', 'Yes'])
StreamingMovies = st.selectbox('Streaming Movies', ['No', 'Yes'])
PaperlessBilling = st.selectbox('Paperless Billing', ['No', 'Yes'])
MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, max_value=150.0, step=0.1)
TotalCharges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, step=0.1)
InternetService = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
PaymentMethod = st.selectbox('Payment Method', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])

# Convert inputs to the format used in training
data = {
    'gender': 1 if gender == 'Female' else 0,
    'SeniorCitizen': SeniorCitizen,
    'Partner': 1 if Partner == 'Yes' else 0,
    'Dependents': 1 if Dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if PhoneService == 'Yes' else 0,
    'MultipleLines': 1 if MultipleLines == 'Yes' else 0,
    'OnlineSecurity': 1 if OnlineSecurity == 'Yes' else 0,
    'OnlineBackup': 1 if OnlineBackup == 'Yes' else 0,
    'DeviceProtection': 1 if DeviceProtection == 'Yes' else 0,
    'TechSupport': 1 if TechSupport == 'Yes' else 0,
    'StreamingTV': 1 if StreamingTV == 'Yes' else 0,
    'StreamingMovies': 1 if StreamingMovies == 'Yes' else 0,
    'PaperlessBilling': 1 if PaperlessBilling == 'Yes' else 0,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,
    'InternetService_DSL': 1 if InternetService == 'DSL' else 0,
    'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
    'InternetService_No': 1 if InternetService == 'No' else 0,
    'Contract_Month-to-month': 1 if Contract == 'Month-to-month' else 0,
    'Contract_One year': 1 if Contract == 'One year' else 0,
    'Contract_Two year': 1 if Contract == 'Two year' else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if PaymentMethod == 'Bank transfer (automatic)' else 0,
    'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0
}

# Convert input data to DataFrame
input_data = pd.DataFrame([data])

# Scale the numerical columns
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

# Predict churn probability
if st.button('Predict'):
    prediction = model.predict(input_data)
    churn_prob = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.write(f"Prediction: Churn with a probability of {churn_prob:.2f}.")
    else:
        st.write(f"Prediction: Not Churn with a probability of {1 - churn_prob:.2f}.")

