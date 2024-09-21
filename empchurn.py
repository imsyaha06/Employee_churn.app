import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

st.set_page_config(page_title="Employee Churn Prediction", layout="centered")

# Title of the app
st.markdown("<h1 style='color: #4CAF50;'>Employee Churn Prediction App</h1>", unsafe_allow_html=True)

# Sidebar inputs for user features
st.sidebar.markdown("### Enter Employee Details")

# Input fields for each feature
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=18.0, max_value=120.0, value=50.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=8684.0, value=500.0)


# Load dataset to simulate how the model was trained
data = pd.read_csv('churn_data.csv')

# Convert the 'Churn' column to numeric values .
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Select your main column
X = data.drop(['Churn', 'customerID'], axis=1)   
y = data['Churn']  # 1 for 'Yes', 0 for 'No'              

# Apply one-hot encoding to categorical features
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the logistic regression model and train it
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to preprocess user input
def preprocess_user_input(gender, senior_citizen, tenure,
                            contract, payment_method, 
                          monthly_charges, total_charges):
    # Create a dictionary with the user input
    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'tenure': tenure,
        'Contract': contract,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Convert categorical variables into the format expected by the model (one-hot encoding)
    user_input_df = pd.DataFrame([user_data])
    user_input_df = pd.get_dummies(user_input_df)
    
    # Align user input with training data structure by adding missing columns with zeros
    user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)
    
    # Apply the same scaling to user input
    user_input_scaled = scaler.transform(user_input_df)
    
    return user_input_scaled

# Determine the optimal probability threshold based on the best F1 score
def get_optimal_threshold(X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]  # Get probability estimates
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]  # Choose threshold with the best F1 score
    return optimal_threshold

# Predict button
if st.sidebar.button("Predict"):
    # Preprocess user input
    user_input = preprocess_user_input(gender, senior_citizen, tenure,  
                                       contract, payment_method, monthly_charges, total_charges)

    # Make the prediction
    optimal_threshold = get_optimal_threshold(X_test, y_test)
    user_prob = model.predict_proba(user_input)[:, 1]  # Get the predicted probability for 'Yes' (1)
    prediction = user_prob >= optimal_threshold
    
    # Display the prediction result
    if prediction:
        st.markdown("<h3 style='color: #F44336;'>The employee is likely to leave the company.</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color: #4CAF50;'>The employee is likely to stay in the company.</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h4 style='color: #2196F3;'>Please enter employee details in the sidebar and press 'Predict'.</h4>", unsafe_allow_html=True)

 with open('churn_pickle', 'rb') as f:
    model = pickle.load(f)

