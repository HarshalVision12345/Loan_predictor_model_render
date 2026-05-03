import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -----------------------------
# 1. Load Model and Columns
# -----------------------------
# This part ensures the 'columns' variable is defined globally
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("columns.pkl", "rb") as f:
        columns = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Error: 'model.pkl' or 'columns.pkl' not found. Please run your training script first.")
    st.stop()  # Stops the app if files are missing

# -----------------------------
# 2. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Loan Predictor", layout="centered")
st.title("🏦 Loan Prediction App")
st.write("Fill in the details below to check loan eligibility.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["No", "Yes"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

with col2:
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    income = st.number_input("Applicant Income", min_value=0, value=5000)
    co_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amt = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)
    loan_term = st.selectbox("Loan Term (Days)", [360, 180, 120, 60, 36])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])

# -----------------------------
# 3. Prediction Logic
# -----------------------------
if st.button("Predict Loan Approval"):
    # Create a dictionary and manually Encode the categories 
    # (Matches your LabelEncoder logic from training)
    input_dict = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": 3 if dependents == "3+" else int(dependents),
        "Education": 0 if education == "Graduate" else 1,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan_amt,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": 0 if property_area == "Rural" else (1 if property_area == "Semiurban" else 2)
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # FIX: Reindex ensures columns are in the exact order the model expects
    # Uses the 'columns' variable loaded at the top
    df_final = df.reindex(columns=columns, fill_value=0)

    # Make prediction
    prediction = model.predict(df_final)[0]

    # Display Result
    st.markdown("---")
    if prediction == 1:
        st.success("🎉 **Congratulations!** The loan is likely to be **Approved**.")
    else:
        st.error("⚠️ **Sorry**, the loan is likely to be **Rejected**.")




