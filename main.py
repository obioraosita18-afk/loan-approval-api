"""
Loan Approval Prediction API
Author: Obiora Osita
"""

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(
    title="🏦 Loan Approval Prediction API",
    description="Predicts whether a loan application will be Approved or Rejected.",
    version="1.0.0",
)

model  = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")

class LoanApplication(BaseModel):
    Gender: Literal["Male", "Female"] = Field(..., example="Male")
    Married: Literal["Yes", "No"] = Field(..., example="Yes")
    Dependents: Literal["0", "1", "2", "3+"] = Field(..., example="1")
    Education: Literal["Graduate", "Not Graduate"] = Field(..., example="Graduate")
    Self_Employed: Literal["Yes", "No"] = Field(..., example="No")
    ApplicantIncome: float = Field(..., example=5000)
    CoapplicantIncome: float = Field(..., example=1500)
    LoanAmount: float = Field(..., example=120)
    Loan_Amount_Term: float = Field(..., example=360)
    Credit_History: float = Field(..., example=1.0)
    Property_Area: Literal["Urban", "Semiurban", "Rural"] = Field(..., example="Urban")

    class Config:
        json_schema_extra = {
            "example": {
                "Gender": "Male", "Married": "Yes", "Dependents": "1",
                "Education": "Graduate", "Self_Employed": "No",
                "ApplicantIncome": 5000, "CoapplicantIncome": 1500,
                "LoanAmount": 120, "Loan_Amount_Term": 360,
                "Credit_History": 1.0, "Property_Area": "Urban"
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    prediction_label: int
    probability_approved: float
    probability_rejected: float

GENDER_MAP     = {"Female": 0, "Male": 1}
MARRIED_MAP    = {"No": 0, "Yes": 1}
DEPENDENTS_MAP = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP  = {"Graduate": 0, "Not Graduate": 1}
SELF_EMP_MAP   = {"No": 0, "Yes": 1}
PROPERTY_MAP   = {"Rural": 0, "Semiurban": 1, "Urban": 2}

def encode_and_engineer(d: LoanApplication) -> np.ndarray:
    applicant_inc   = d.ApplicantIncome
    coapplicant_inc = d.CoapplicantIncome
    loan_amount     = d.LoanAmount
    total_income    = applicant_inc + coapplicant_inc

    features = np.array([[
        0,
        GENDER_MAP[d.Gender],
        MARRIED_MAP[d.Married],
        DEPENDENTS_MAP[d.Dependents],
        EDUCATION_MAP[d.Education],
        SELF_EMP_MAP[d.Self_Employed],
        applicant_inc,
        coapplicant_inc,
        loan_amount,
        d.Loan_Amount_Term,
        d.Credit_History,
        PROPERTY_MAP[d.Property_Area],
        np.log(applicant_inc + 1),
        np.log(loan_amount + 1),
        total_income,
        np.log(total_income + 1),
    ]])
    return scaler.transform(features)

@app.get("/", tags=["Health"])
def root():
    return {"message": "🏦 Loan Approval API is running!", "docs": "/docs"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(application: LoanApplication):
    try:
        features   = encode_and_engineer(application)
        pred_label = int(model.predict(features)[0])
        proba      = model.predict_proba(features)[0]
        return PredictionResponse(
            prediction="Approved" if pred_label == 1 else "Rejected",
            prediction_label=pred_label,
            probability_approved=round(float(proba[1]), 4),
            probability_rejected=round(float(proba[0]), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
