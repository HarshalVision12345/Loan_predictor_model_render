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




# import streamlit as st
# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Training phase (runs once)
# -----------------------------
# @st.cache_resource
# def train_model():
#     # Load dataset
#     data = pd.read_csv("loan_dataset.csv")

#     # Drop Loan_ID if present
#     if "Loan_ID" in data.columns:
#         data = data.drop("Loan_ID", axis=1)

#     # One-hot encode categorical variables
#     data = pd.get_dummies(data)

#     # Split features and target
#     X = data.drop("Loan_Status", axis=1)   # adjust target column name if different
#     y = data["Loan_Status"]

#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train model
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)

#     # Save model and columns
#     pickle.dump(model, open("model.pkl", "wb"))
#     pickle.dump(X.columns, open("columns.pkl", "wb"))

#     return model, X.columns

# # Load or train model
# try:
#     model = pickle.load(open("model.pkl", "rb"))
#     columns = pickle.load(open("columns.pkl", "rb"))
# except:
#     model, columns = train_model()

# # -----------------------------
# # 2. Streamlit UI
# # -----------------------------
# st.title("🏦 Loan Prediction App")

# # User input form
# gender = st.selectbox("Gender", ["Male", "Female"])
# married = st.selectbox("Married", ["Yes", "No"])
# education = st.selectbox("Education", ["Graduate", "Not Graduate"])
# self_employed = st.selectbox("Self Employed", ["Yes", "No"])
# applicant_income = st.number_input("Applicant Income", min_value=0)
# coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
# loan_amount = st.number_input("Loan Amount", min_value=0)
# loan_term = st.selectbox("Loan Amount Term", [360, 180, 120, 60])
# credit_history = st.selectbox("Credit History", [1.0, 0.0])
# property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# # Collect input
# input_data = {
#     "Gender": gender,
#     "Married": married,
#     "Education": education,
#     "Dependents": "0",
#     "Self_Employed": self_employed,
#     "ApplicantIncome": applicant_income,
#     "CoapplicantIncome": coapplicant_income,
#     "LoanAmount": loan_amount,
#     "Loan_Amount_Term": loan_term,
#     "Credit_History": credit_history,
#     "Property_Area": property_area
# }

# df = pd.DataFrame([input_data])

# # Apply same preprocessing
# df = pd.get_dummies(df)
# df = df.reindex(columns=columns, fill_value=0)

# # Prediction button
# if st.button("Predict Loan Approval"):
#     prediction = model.predict(df)[0]
#     st.success(f"✅ Loan Prediction: {prediction}")