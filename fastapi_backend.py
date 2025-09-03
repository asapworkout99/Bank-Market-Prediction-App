from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import joblib
import json
import time
import os  # Add this missing import
from datetime import datetime

app = FastAPI(title="Bank Marketing Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BankMarketingModel:
    def __init__(self, models_path="./models/"):
        self.models_path = models_path
        self.models = {}
        self.encoders = {}
        self.selected_features = []
        self.metadata = {}
        self.loaded = False
        
        try:
            self.load_models()
            self.loaded = True
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.loaded = False
    
    def load_models(self):
        """Load all model components"""
        model_files = {
            'xgb': 'xgb_model.joblib',
            'lgb': 'lgb_model.joblib'
        }
        
        for name, filename in model_files.items():
            path = os.path.join(self.models_path, filename)
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"Loaded {name} model")
        
        ensemble_path = os.path.join(self.models_path, "ensemble_model.joblib")
        if os.path.exists(ensemble_path):
            self.models['ensemble'] = joblib.load(ensemble_path)
            print("Loaded ensemble model")
        
        encoders_path = os.path.join(self.models_path, "encoders.joblib")
        if os.path.exists(encoders_path):
            self.encoders = joblib.load(encoders_path)
            print("Loaded encoders")
        
        features_path = os.path.join(self.models_path, "selected_features.json")
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
            print(f"Loaded {len(self.selected_features)} features")
        
        metadata_path = os.path.join(self.models_path, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print("Loaded model metadata")
    
    def create_features(self, data):
        """Apply the same feature engineering as training"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100],
                                labels=['young', 'adult', 'middle', 'senior', 'elderly'])
        
        # Balance features
        df['balance_positive'] = (df['balance'] > 0).astype(int)
        df['balance_negative'] = (df['balance'] < 0).astype(int)
        df['balance_zero'] = (df['balance'] == 0).astype(int)
        df['balance_log'] = np.log1p(np.abs(df['balance']))
        
        # Duration features
        df['duration_long'] = (df['duration'] > 300).astype(int)
        df['duration_short'] = (df['duration'] < 60).astype(int)
        df['duration_log'] = np.log1p(df['duration'])
        
        # Campaign features
        df['campaign_high'] = (df['campaign'] > 3).astype(int)
        df['campaign_single'] = (df['campaign'] == 1).astype(int)
        
        # Previous contact features
        df['has_previous'] = (df['previous'] > 0).astype(int)
        df['pdays_contacted'] = (df['pdays'] != -1).astype(int)
        
        # Education ordinal mapping
        education_order = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}
        df['education_ord'] = df['education'].map(education_order)
        
        # Month seasonality
        month_season = {
            'dec': 'winter', 'jan': 'winter', 'feb': 'winter',
            'mar': 'spring', 'apr': 'spring', 'may': 'spring',
            'jun': 'summer', 'jul': 'summer', 'aug': 'summer',
            'sep': 'autumn', 'oct': 'autumn', 'nov': 'autumn'
        }
        df['season'] = df['month'].map(month_season)
        
        # Job success potential
        job_success_rate = {
            'management': 'high', 'technician': 'medium', 'entrepreneur': 'high',
            'blue-collar': 'low', 'unknown': 'low', 'retired': 'medium',
            'admin.': 'medium', 'services': 'low', 'self-employed': 'medium',
            'unemployed': 'low', 'housemaid': 'low', 'student': 'medium'
        }
        df['job_success_potential'] = df['job'].map(job_success_rate).fillna('low')
        
        # Interaction features
        df['young_student'] = ((df['age'] < 25) & (df['job'] == 'student')).astype(int)
        df['retired_senior'] = ((df['age'] > 60) & (df['job'] == 'retired')).astype(int)
        df['high_balance_management'] = ((df['balance'] > 1000) & (df['job'] == 'management')).astype(int)
        df['previous_success'] = ((df['poutcome'] == 'success') & (df['previous'] > 0)).astype(int)
        df['previous_failure'] = ((df['poutcome'] == 'failure') & (df['previous'] > 0)).astype(int)
        
        # Z-scores for numerical features
        for col in ['age', 'balance', 'duration']:
            if df[col].std() != 0:
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
            else:
                df[f'{col}_zscore'] = 0
        
        return df
    
    def preprocess_data(self, data):
        """Preprocess data for prediction"""
        if not self.loaded:
            raise ValueError("Models not loaded properly")
        
        df = self.create_features(data)
        
        # Apply encoders to categorical columns
        for col, encoder in self.encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except Exception as e:
                    print(f"Warning: Could not encode column {col}: {e}")
                    df[col] = 0  # Default value for encoding errors
        
        # Select only the features used during training
        missing_features = [f for f in self.selected_features if f not in df.columns]
        if missing_features:
            for feature in missing_features:
                df[feature] = 0  # Add missing features with default value
        
        return df[self.selected_features]
    
    def predict(self, data):
        """Make prediction using the ensemble model"""
        if not self.loaded:
            return {"error": "Models not loaded"}
        
        try:
            processed_data = self.preprocess_data(data)
            
            # Get ensemble prediction
            if 'ensemble' in self.models:
                prob = self.models['ensemble'].predict_proba(processed_data)[0][1]
            elif 'xgb' in self.models:
                prob = self.models['xgb'].predict_proba(processed_data)[0][1]
            else:
                return {"error": "No models available for prediction"}
            
            # Risk categorization
            if prob > 0.7:
                risk = "High Likelihood"
            elif prob > 0.4:
                risk = "Medium Likelihood" 
            else:
                risk = "Low Likelihood"
            
            # Generate insights based on data
            insights = self._generate_insights(data, prob)
            
            return {
                "subscription_probability": float(prob),
                "likelihood": risk,
                "confidence_score": int(prob * 100),
                "model_confidence": "High" if prob > 0.8 or prob < 0.2 else "Medium",
                "key_factors": insights,
                "model_version": self.metadata.get("model_version", "2.0.0")
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _generate_insights(self, data, probability):
        """Generate insights based on customer data"""
        insights = []
        
        # Age-based insights
        age = data.get('age', 0)
        if age < 30:
            insights.append("Young demographic - typically tech-savvy")
        elif age > 60:
            insights.append("Senior demographic - may prefer traditional banking")
        
        # Balance insights
        balance = data.get('balance', 0)
        if balance > 2000:
            insights.append("High account balance - good financial standing")
        elif balance < 0:
            insights.append("Negative balance - potential financial stress")
        
        # Duration insights
        duration = data.get('duration', 0)
        if duration > 300:
            insights.append("Long call duration - engaged customer")
        elif duration < 60:
            insights.append("Short call - limited engagement")
        
        # Job insights
        job = data.get('job', '')
        if job in ['management', 'entrepreneur']:
            insights.append("Professional role - higher income potential")
        elif job == 'retired':
            insights.append("Retired - stable but limited income")
        
        # Previous campaign insights
        previous = data.get('previous', 0)
        if previous > 0:
            poutcome = data.get('poutcome', '')
            if poutcome == 'success':
                insights.append("Previously successful - high conversion potential")
            elif poutcome == 'failure':
                insights.append("Previous campaign failed - may need different approach")
        
        return insights[:5]  # Return top 5 insights

# Initialize the model
model = BankMarketingModel()

# Pydantic models for API
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    month: str

class BatchRequest(BaseModel):
    customers: List[Dict[str, Any]]

# API endpoints
@app.get("/")
async def root():
    return {
        "status": "API is running", 
        "model_loaded": model.loaded, 
        "model_version": model.metadata.get("model_version", "2.0.0"),
        "available_models": list(model.models.keys()) if model.loaded else [],
        "feature_count": len(model.selected_features) if model.loaded else 0
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": model.loaded,
        "models_available": list(model.models.keys()) if model.loaded else []
    }

@app.post("/predict")
async def predict_subscription(customer: CustomerData):
    """Predict subscription probability for a single customer"""
    try:
        prediction = model.predict(customer.dict())
        if "error" in prediction:
            raise HTTPException(status_code=500, detail=prediction["error"])
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def batch_predict(request: BatchRequest):
    """Predict subscription probability for multiple customers"""
    try:
        predictions = []
        for customer in request.customers:
            pred = model.predict(customer)
            predictions.append(pred)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get detailed model information"""
    return {
        "metadata": model.metadata,
        "selected_features": model.selected_features[:10],  # Show first 10
        "total_features": len(model.selected_features),
        "encoders_available": list(model.encoders.keys()) if model.loaded else [],
        "models_loaded": list(model.models.keys()) if model.loaded else []
    }

@app.get("/model/features")
async def get_feature_list():
    """Get complete list of features used by the model"""
    if not model.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "features": model.selected_features,
        "feature_count": len(model.selected_features)
    }

# Sample data endpoint for testing
@app.get("/sample-data")
async def get_sample_data():
    """Get sample customer data for testing"""
    return {
        "sample_customer": {
            "age": 35,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "balance": 1500.5,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "duration": 180,
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
            "month": "may"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Bank Marketing Prediction API...")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)