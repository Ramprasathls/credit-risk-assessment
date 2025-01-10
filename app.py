import streamlit as st
import pandas as pd
import joblib

# Load the saved Random Forest pipeline
pipeline = joblib.load("E:/credit_risk_assessment/credit-risk-assessment/models/weighted_random_forest_pipeline.pkl")

# Streamlit app title
st.title("Credit Risk Prediction App")
st.write(
    """
    This app predicts whether a loan applicant is at high risk of defaulting based on their details.
    """
)

# Input fields for user data
st.header("Enter Applicant Details")
person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income ($)", min_value=0, value=50000)
person_home_ownership = st.selectbox(
    "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
)
person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=3.0)
loan_intent = st.selectbox(
    "Loan Intent",
    ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
)
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount ($)", min_value=1000, value=15000)
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=12.5)
loan_percent_income = st.number_input(
    "Loan as % of Income", min_value=0.0, max_value=1.0, value=0.3
)
cb_person_default_on_file = st.selectbox("Default History", ["Y", "N"])
cb_person_cred_hist_length = st.number_input(
    "Credit History Length (years)", min_value=0, value=6
)

# Collect the inputs into a DataFrame
input_data = pd.DataFrame(
    [
        {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }
    ]
)

# Prediction button
if st.button("Predict Credit Risk"):
    # Preprocess and predict
    prediction = pipeline["model"].predict(pipeline["preprocessor"].transform(input_data))
    
    # Display the result
    if prediction[0] == 1:
        st.error("High Risk of Default! 🚨")
    else:
        st.success("Low Risk of Default! ✅")
