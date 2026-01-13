import os
import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stock_prediction_model.keras")

@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH, compile=False)

model = load_lstm_model()
scaler = MinMaxScaler(feature_range=(0, 1))

st.title("Stock Price Prediction (ML Service)")
st.write("Paste at least **120 closing prices** (comma separated)")

prices_input = st.text_area("Paste closing prices (comma separated)")

if st.button("Predict"):
    try:
        prices = np.array(
            [float(x.strip()) for x in prices_input.split(",") if x.strip()]
        ).reshape(-1, 1)

        if len(prices) < 120:
            st.error("Please enter at least 120 prices for prediction")
            st.stop()

        scaled_data = scaler.fit_transform(prices)

        x_test = []
        for i in range(100, len(scaled_data)):
            x_test.append(scaled_data[i-100:i])

        x_test = np.array(x_test)

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        st.success("Prediction completed successfully")
        st.line_chart(predictions.flatten())

    except Exception as e:
        st.error(f"Error: {e}")
