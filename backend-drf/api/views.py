from django.shortcuts import render
from rest_framework.views import APIView
from .serializers import StockPredictionSerializer
from rest_framework import status
from rest_framework.response import Response
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from django.conf import settings
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import AllowAny

from .utils import save_plot
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score

class CsrfExemptSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return

class StockPredictionAPIView(APIView):

    authentication_classes = (CsrfExemptSessionAuthentication,)
    permission_classes = (AllowAny,)

    def post(self, request):
        serializer = StockPredictionSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )

        ticker = serializer.validated_data['ticker'].upper()

        # -------------------------------
        # 1️⃣ Fetch stock data safely
        # -------------------------------
        try:
            now = datetime.now()
            start = datetime(now.year - 10, now.month, now.day)

            df = yf.download(ticker, start, now, progress=False)

            if df.empty:
                return Response(
                    {"error": f"No data found for ticker '{ticker}'"},
                    status=status.HTTP_404_NOT_FOUND
                )

        except Exception as e:
            return Response(
                {"error": "Failed to fetch stock data", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        df = df.reset_index()
        plt.switch_backend('AGG')

        # -------------------------------
        # 2️⃣ Plots
        # -------------------------------
        plt.figure(figsize=(12, 5))
        plt.plot(df.Close, label='Closing Price')
        plt.legend()
        plot_img = save_plot(f'{ticker}_plot.png')

        ma100 = df.Close.rolling(100).mean()
        plt.figure(figsize=(12, 5))
        plt.plot(df.Close)
        plt.plot(ma100, 'r')
        plot_100_dma = save_plot(f'{ticker}_100_dma.png')

        ma200 = df.Close.rolling(200).mean()
        plt.figure(figsize=(12, 5))
        plt.plot(df.Close)
        plt.plot(ma100, 'r')
        plt.plot(ma200, 'g')
        plot_200_dma = save_plot(f'{ticker}_200_dma.png')

        # -------------------------------
        # 3️⃣ Prepare ML data
        # -------------------------------
        data_training = pd.DataFrame(df.Close[:int(len(df) * 0.7)])
        data_testing = pd.DataFrame(df.Close[int(len(df) * 0.7):])

        scaler = MinMaxScaler(feature_range=(0, 1))

        model_path = os.path.join(settings.BASE_DIR, 'stock_prediction_model.keras')
        if not os.path.exists(model_path):
            return Response(
                {"error": "ML model not found on server"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        model = load_model(model_path)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing])
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # -------------------------------
        # 4️⃣ Prediction
        # -------------------------------
        y_predicted = model.predict(x_test)

        y_predicted = scaler.inverse_transform(y_predicted.reshape(-1, 1)).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        plt.figure(figsize=(12, 5))
        plt.plot(y_test, label='Actual')
        plt.plot(y_predicted, 'r', label='Predicted')
        plt.legend()
        plot_prediction = save_plot(f'{ticker}_prediction.png')

        # -------------------------------
        # 5️⃣ Metrics
        # -------------------------------
        mse = mean_squared_error(y_test, y_predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_predicted)

        # -------------------------------
        # 6️⃣ Final Response
        # -------------------------------
        return Response(
            {
                "status": "success",
                "ticker": ticker,
                "plots": {
                    "price": plot_img,
                    "dma_100": plot_100_dma,
                    "dma_200": plot_200_dma,
                    "prediction": plot_prediction,
                },
                "metrics": {
                    "mse": round(float(mse), 4),
                    "rmse": round(float(rmse), 4),
                    "r2": round(float(r2), 4),
                }
            },
            status=status.HTTP_200_OK
        )
