import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# Load model only once (important for performance)
# --------------------------------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("stock_prediction_model.keras")

model = load_lstm_model()
scaler = MinMaxScaler(feature_range=(0, 1))

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Stock Price Prediction (ML Service)")
st.write("Paste at least **120 closing prices** (comma separated)")

prices_input = st.text_area(
    "Paste closing prices (comma separated)",
    placeholder="100,101,102,103,..."
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict"):

    if not prices_input.strip():
        st.error("Input cannot be empty")
        st.stop()

    # Convert input to float safely
    try:
        prices = [float(x.strip()) for x in prices_input.split(",")]
    except ValueError:
        st.error("Please enter only numeric values separated by commas")
        st.stop()

    # Validate length for LSTM
    if len(prices) < 120:
        st.error("Please enter at least 120 prices for prediction")
        st.stop()

    prices = np.array(prices).reshape(-1, 1)

    # Scale data
    scaled_data = scaler.fit_transform(prices)

    # Create sequences (100 timesteps)
    x_test = []
    for i in range(100, len(scaled_data)):
        x_test.append(scaled_data[i-100:i])

    x_test = np.array(x_test)

    # Safety check
    if x_test.shape[0] == 0:
        st.error("Not enough data after processing")
        st.stop()

    # Predict (disable verbose to avoid math domain error)
    predictions = model.predict(x_test, verbose=0)

    # Inverse scaling
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Output
    st.success("Prediction completed successfully")
    st.line_chart(predictions)
    st.json({
        "predictions": predictions.flatten().tolist()
    })
