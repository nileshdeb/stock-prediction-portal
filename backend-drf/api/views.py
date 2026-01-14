from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import AllowAny

from .serializers import StockPredictionSerializer
from .utils import save_plot

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import os

from django.conf import settings
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score


# -------------------------------
# Disable CSRF (API-only)
# -------------------------------
class CsrfExemptSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return


class StockPredictionAPIView(APIView):
    authentication_classes = (CsrfExemptSessionAuthentication,)
    permission_classes = (AllowAny,)

    def post(self, request):
        serializer = StockPredictionSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        ticker = serializer.validated_data["ticker"].upper()

        # -------------------------------
        # 1️⃣ Fetch stock data (Railway safe)
        # -------------------------------
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="10y", interval="1d")

            if df.empty:
                return Response(
                    {"error": f"No data found for ticker '{ticker}'"},
                    status=status.HTTP_404_NOT_FOUND
                )

        except Exception as e:
            return Response(
                {"error": "Stock data fetch failed", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        df.reset_index(inplace=True)

        if "Close" not in df.columns:
            return Response(
                {"error": "Close price not available for this ticker"},
                status=status.HTTP_400_BAD_REQUEST
            )

        prices = df["Close"]

        plt.switch_backend("AGG")

        # -------------------------------
        # 2️⃣ Price Plot
        # -------------------------------
        plt.figure(figsize=(12, 5))
        plt.plot(prices, label="Closing Price")
        plt.legend()
        plot_price = save_plot(f"{ticker}_price.png")
        plt.close()

        # -------------------------------
        # 3️⃣ Moving Averages
        # -------------------------------
        ma100 = prices.rolling(100).mean()
        ma200 = prices.rolling(200).mean()

        plt.figure(figsize=(12, 5))
        plt.plot(prices)
        plt.plot(ma100, "r", label="100 DMA")
        plt.legend()
        plot_100 = save_plot(f"{ticker}_100_dma.png")
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(prices)
        plt.plot(ma100, "r", label="100 DMA")
        plt.plot(ma200, "g", label="200 DMA")
        plt.legend()
        plot_200 = save_plot(f"{ticker}_200_dma.png")
        plt.close()

        # -------------------------------
        # 4️⃣ Prepare ML Data
        # -------------------------------
        split = int(len(prices) * 0.7)
        train_data = prices[:split]
        test_data = prices[split:]

        if len(train_data) < 120 or len(test_data) < 30:
            return Response(
                {"error": "Not enough data for prediction"},
                status=status.HTTP_400_BAD_REQUEST
            )

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))

        past_100 = train_scaled[-100:]
        test_scaled = scaler.transform(test_data.values.reshape(-1, 1))
        final_input = np.vstack((past_100, test_scaled))

        x_test, y_test = [], []
        for i in range(100, final_input.shape[0]):
            x_test.append(final_input[i - 100:i])
            y_test.append(final_input[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        if len(x_test) == 0:
            return Response(
                {"error": "Insufficient data for prediction window"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # -------------------------------
        # 5️⃣ Load Model
        # -------------------------------
        model_path = os.path.join(settings.BASE_DIR, "stock_prediction_model.keras")

        if not os.path.exists(model_path):
            return Response(
                {"error": "ML model not found on server"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        model = load_model(model_path)

        # -------------------------------
        # 6️⃣ Prediction
        # -------------------------------
        y_predicted = model.predict(x_test)
        y_predicted = scaler.inverse_transform(y_predicted).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        plt.figure(figsize=(12, 5))
        plt.plot(y_test, label="Actual")
        plt.plot(y_predicted, "r", label="Predicted")
        plt.legend()
        plot_prediction = save_plot(f"{ticker}_prediction.png")
        plt.close()

        # -------------------------------
        # 7️⃣ Metrics
        # -------------------------------
        mse = mean_squared_error(y_test, y_predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_predicted)

        # -------------------------------
        # 8️⃣ Final Response
        # -------------------------------
        return Response(
            {
                "status": "success",
                "ticker": ticker,
                "plots": {
                    "price": plot_price,
                    "dma_100": plot_100,
                    "dma_200": plot_200,
                    "prediction": plot_prediction,
                },
                "metrics": {
                    "mse": round(float(mse), 4),
                    "rmse": round(float(rmse), 4),
                    "r2": round(float(r2), 4),
                },
            },
            status=status.HTTP_200_OK
        )
