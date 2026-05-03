from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os

# 1. Use absolute paths to avoid FileNotFoundError
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

# Load model and columns safely
model = load_pickle("model.pkl")
columns = load_pickle("columns.pkl")

app = FastAPI(title="Loan Prediction API")

class LoanInput(BaseModel):
    Gender: str
    Married: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: int
    Credit_History: float
    Property_Area: str

@app.post("/predict")
def predict_loan(data: LoanInput):
    # 2. Use model_dump() for Pydantic v2
    input_data = data.model_dump()
    
    # 3. Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # 4. Handle Categorical Encoding
    df = pd.get_dummies(df)
    
    # 5. Align with training columns (Crucial step!)
    df = df.reindex(columns=columns, fill_value=0)
    
    # 6. Predict
    prediction = model.predict(df)[0]
    
    # Return result (Converting to native Python type for JSON)
    result = "Approved" if prediction == 1 else "Rejected"
    return {"loan_status": result, "raw_prediction": int(prediction)}