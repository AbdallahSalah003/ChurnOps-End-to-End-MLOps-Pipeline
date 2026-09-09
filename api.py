from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.logger import get_logger

logger = get_logger(__name__)

MODEL_PATH = Path("data/model.joblib")

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        models["model"] = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning("Model not found at %s", MODEL_PATH)
    yield
    models.clear()


app = FastAPI(title="Telco Churn Prediction API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    Partner: int
    Dependents: int
    PhoneService: int
    PaperlessBilling: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    MultipleLines: int
    gender_Female: int
    gender_Male: int
    InternetService_DSL: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Bank_transfer_automatic: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int
    tenure_group: int
    avg_monthly_charge: float
    service_count: int
    monthly_charge_x_tenure: float


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": "model" in models,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = [
        request.SeniorCitizen,
        request.tenure,
        request.MonthlyCharges,
        request.TotalCharges,
        request.Partner,
        request.Dependents,
        request.PhoneService,
        request.PaperlessBilling,
        request.OnlineSecurity,
        request.OnlineBackup,
        request.DeviceProtection,
        request.TechSupport,
        request.StreamingTV,
        request.StreamingMovies,
        request.MultipleLines,
        request.gender_Female,
        request.gender_Male,
        request.InternetService_DSL,
        request.InternetService_Fiber_optic,
        request.InternetService_No,
        request.Contract_Month_to_month,
        request.Contract_One_year,
        request.Contract_Two_year,
        request.PaymentMethod_Bank_transfer_automatic,
        request.PaymentMethod_Credit_card_automatic,
        request.PaymentMethod_Electronic_check,
        request.PaymentMethod_Mailed_check,
        request.tenure_group,
        request.avg_monthly_charge,
        request.service_count,
        request.monthly_charge_x_tenure,
    ]

    X = np.array(features).reshape(1, -1)

    model = models["model"]
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    logger.info("Prediction: %d (prob=%.4f)", prediction, probability)

    return PredictionResponse(
        prediction=prediction,
        probability=probability,
    )
