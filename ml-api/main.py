from fastapi import FastAPI, HTTPException
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os

app = FastAPI(title="Stock Prediction ML API")

# Load model ONCE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stock_prediction_model.keras")

model = load_model(MODEL_PATH, compile=False)
scaler = MinMaxScaler(feature_range=(0, 1))


@app.get("/")
def health_check():
    return {"status": "ML API is running"}


@app.get("/predict")
def predict(ticker: str):
    now = datetime.now()
    start = datetime(now.year - 10, now.month, now.day)

    df = yf.download(ticker, start=start, end=now, progress=False)

    if df.empty or len(df) < 120:
        raise HTTPException(
            status_code=400,
            detail="Not enough stock data to make prediction"
        )

    data = df[['Close']]
    scaled = scaler.fit_transform(data)

    X = []
    for i in range(100, len(scaled)):
        X.append(scaled[i - 100:i])

    X = np.array(X)

    preds = model.predict(X)
    preds = scaler.inverse_transform(preds)

    return {
        "ticker": ticker,
        "last_close": float(data.iloc[-1, 0]),
        "predicted_next": float(preds[-1, 0])
    }
