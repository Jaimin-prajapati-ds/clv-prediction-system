"""
FastAPI REST API for CLV Prediction
===================================

Production-ready API for predicting customer lifetime value.

Author: Jaimin Prajapati
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="CLV Prediction API",
    description="REST API for Customer Lifetime Value prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    recency: int = Field(..., ge=0, description="Days since last purchase")
    frequency: int = Field(..., ge=1, description="Total number of transactions")
    monetary: float = Field(..., ge=0, description="Total spending amount in dollars")
    avg_purchase_value: Optional[float] = Field(None, description="Average purchase value")
    customer_lifetime_days: Optional[int] = Field(None, description="Days since first purchase")


class CLVPrediction(BaseModel):
    customer_id: str
    predicted_clv: float
    confidence_interval: List[float]
    segment: str
    recommendation: str
    risk_score: float


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# Helper Functions
def calculate_clv_score(features: CustomerFeatures) -> float:
    avg_purchase = features.avg_purchase_value or (features.monetary / features.frequency)
    lifetime_days = features.customer_lifetime_days or 365
    
    base_score = features.monetary * (1 + features.frequency * 0.15)
    recency_factor = max(0, 1 - (features.recency / 365))
    lifetime_factor = min(1.5, lifetime_days / 365)
    
    clv_score = base_score * recency_factor * lifetime_factor
    
    return round(clv_score, 2)


def assign_customer_segment(recency: int, frequency: int, monetary: float) -> str:
    if recency <= 30 and frequency >= 10 and monetary >= 1000:
        return "Champions"
    elif recency <= 60 and frequency >= 5 and monetary >= 500:
        return "Loyal Customers"
    elif recency <= 90 and frequency >= 3:
        return "Potential Loyalists"
    elif recency <= 30 and frequency <= 2:
        return "Recent Customers"
    elif recency > 180 and frequency >= 5:
        return "At Risk"
    elif recency > 180 and frequency >= 3 and monetary >= 500:
        return "Cannot Lose Them"
    elif recency > 90 and recency <= 180:
        return "Hibernating"
    else:
        return "Lost"


def get_segment_recommendation(segment: str) -> str:
    recommendations = {
        "Champions": "Reward with exclusive benefits and VIP treatment.",
        "Loyal Customers": "Upsell premium products. Request reviews.",
        "Potential Loyalists": "Offer loyalty program membership.",
        "Recent Customers": "Provide excellent onboarding experience.",
        "At Risk": "Launch win-back campaign with special discounts.",
        "Cannot Lose Them": "URGENT: Aggressive retention needed.",
        "Hibernating": "Re-engagement campaign with compelling offers.",
        "Lost": "Minimal investment. Monitor for return."
    }
    return recommendations.get(segment, "Monitor customer behavior.")


def calculate_risk_score(recency: int, frequency: int) -> float:
    recency_risk = min(recency / 365, 1.0)
    frequency_risk = max(0, 1 - (frequency / 20))
    risk_score = (recency_risk * 0.6 + frequency_risk * 0.4)
    return round(risk_score, 3)


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict_clv", response_model=CLVPrediction)
async def predict_clv(features: CustomerFeatures):
    try:
        predicted_clv = calculate_clv_score(features)
        
        confidence_lower = round(predicted_clv * 0.85, 2)
        confidence_upper = round(predicted_clv * 1.15, 2)
        
        segment = assign_customer_segment(
            features.recency,
            features.frequency,
            features.monetary
        )
        
        recommendation = get_segment_recommendation(segment)
        risk_score = calculate_risk_score(features.recency, features.frequency)
        
        return CLVPrediction(
            customer_id=features.customer_id,
            predicted_clv=predicted_clv,
            confidence_interval=[confidence_lower, confidence_upper],
            segment=segment,
            recommendation=recommendation,
            risk_score=risk_score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=List[CLVPrediction])
async def batch_predict_clv(customers: List[CustomerFeatures]):
    if len(customers) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limit exceeded. Maximum 100 customers per request."
        )
    
    predictions = []
    
    for customer in customers:
        try:
            prediction = await predict_clv(customer)
            predictions.append(prediction)
        except Exception as e:
            print(f"Error predicting for {customer.customer_id}: {str(e)}")
            continue
    
    return predictions


@app.get("/segments")
async def get_segment_info():
    return {
        "segments": [
            {
                "name": "Champions",
                "description": "Best customers with high R, F, and M scores",
                "criteria": "Recency <= 30 days, Frequency >= 10, Monetary >= $1000",
                "priority": "High"
            },
            {
                "name": "Loyal Customers",
                "description": "Regular purchasers with good spending habits",
                "criteria": "Recency <= 60 days, Frequency >= 5, Monetary >= $500",
                "priority": "High"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)