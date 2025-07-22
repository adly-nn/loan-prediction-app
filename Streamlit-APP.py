import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained pipeline
model = joblib.load('final_model.pkl')

st.title("Loan Approval Prediction App")
st.write("This app predicts whether a loan will be approved based on user input features.")

# Input features
gender = st.selectbox("Gender", ["Male", "Female"])
employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Self-Employed"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
income = st.number_input("Income", min_value=0, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
previous_loans = st.number_input("Number of Previous Loans", min_value=0, value=0)
previous_defaults = st.number_input("Number of Previous Defaults", min_value=0, value=0)
loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
loan_purpose = st.selectbox("Loan Purpose", ["Medical", "Travel", "Business", "Education", "Rent"])
bvn_verified = st.number_input("BVN Verified (1 for Yes, 0 for No)", min_value=0, max_value=1, value=1)
guarantor_available = st.number_input("Guarantor (1 for Yes, 0 for No)", min_value=0, max_value=1, value=0)
region = st.selectbox("Region", ["Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa",
    "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger",
    "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara"])
# Create a DataFrame for the input features
input_data = pd.DataFrame({
    'gender': [gender],
    'employment_status': [employment],
    'age': [age],
    'monthly_income': [income],
    'loan_amount': [loan_amount],
    'previous_loans': [previous_loans],
    'previous_defaults': [previous_defaults],
    'loan_term': [loan_term],
    'credit_score': [credit_score],
    'loan_purpose': [loan_purpose],
    'bvn_verified': [bvn_verified],
    'guarantor_available': [guarantor_available],
    'region': [region]

})

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Prediction:", "Approved" if prediction[0] == 1 else "Not Approved")

# Display the input data for debugging
st.subheader("Input Data")
st.write(input_data)

