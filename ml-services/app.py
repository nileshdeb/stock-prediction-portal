import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import json
import sys

st.set_page_config(page_title="Stock Prediction ML Service", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stock_prediction_model.keras")

@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH, compile=False)

model = load_lstm_model()
scaler = MinMaxScaler(feature_range=(0, 1))

st.title("📈 Stock Price Prediction – ML Service")

# -----------------------------
# API MODE (called by Django)
# -----------------------------
query_params = st.experimental_get_query_params()
ticker = query_params.get("ticker", [None])[0]

def predict_from_ticker(ticker):
    now = datetime.now()
    start = datetime(now.year - 10, now.month, now.day)

    df = yf.download(ticker, start=start, end=now, progress=False)

    if df.empty:
        return {"error": "No data found"}

    data = df[['Close']]
    scaled = scaler.fit_transform(data)

    X = []
    for i in range(100, len(scaled)):
        X.append(scaled[i-100:i])

    X = np.array(X)

    preds = model.predict(X)
    preds = scaler.inverse_transform(preds)

    return {
        "ticker": ticker,
        "last_close": float(data.iloc[-1]),
        "predicted_next": float(preds[-1])
    }

if ticker:
    result = predict_from_ticker(ticker)

    # 👇 return raw JSON (no UI)
    st.write(result)
    st.stop()

# -----------------------------
# UI MODE (manual testing)
# -----------------------------
st.subheader("Manual Test")

input_ticker = st.text_input("Enter Stock Ticker", value="AAPL")

if st.button("Predict"):
    result = predict_from_ticker(input_ticker)
    st.json(result)
