import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
label_encoder = LabelEncoder()
scaler = StandardScaler()

model = pickle.load(open('7 logistic_model.pkl','rb'))
df = pd.read_csv("churn_data.csv")

st.title("Employee Churn Prediction")
gender = st.selectbox("Select Gender",options=['Female','Male'])
SeniorCitizen = st.selectbox("Are you a senior citizen?", options=['Yes','No'])
Partner = st.selectbox("Do you have partner?", options=['Yes','No'])
Dependents	 = st.selectbox("Are you dependent on others?", options=['Yes','No'])
tenure = st.text_input("Enter Your tenure in months")
PhoneService = st.selectbox("Do have phone service?",options=['Yes','No'])
MultipleLines = st.selectbox("Do you have mutlilines servics?", options=['Yes','No','no phone service'])
Contract = st.selectbox("Your Contract?",options=['One year','Two year','Month-to_month'])
TotalCharges = st.text_input("Enter your Total charges?")


def prediction(gender,Seniorcitizen,Partner,Dependents,tenure,Phoneservice,multiline,contact,totalcharge):
    data = {
    'gender': [gender],
    'SeniorCitizen': [Seniorcitizen],
    'Partner': [Partner],
    'Dependents': [Dependents],
    'tenure': [tenure],
    'PhoneService': [Phoneservice],
    'MultipleLines': [multiline],
    'Contract': [contact],
    'TotalCharges': [totalcharge]
    }
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)


    # Encode the categorical columns
    categorical_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'Contract']
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    df = scaler.fit_transform(df)

    result = model.predict(df).reshape(1,-1)
    return result[0]


# Create DataFrames
churn_tips_df = pd.DataFrame(churn_tips_data)
retention_tips_df = pd.DataFrame(retention_tips_data)

if st.button("Predict churn or not"):
    result = prediction(gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, Contract,TotalCharges)
    if result == 1:
        st.title("Churn")
         
    else:
        st.title('Not Churn')
         
        
    
