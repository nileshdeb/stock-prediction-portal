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
        try:
            serializer = StockPredictionSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    serializer.errors,
                    status=status.HTTP_400_BAD_REQUEST
                )

            ticker = serializer.validated_data['ticker']

            # -------------------------------
            # 1️⃣ Fetch stock data
            # -------------------------------
            now = datetime.now()
            start = datetime(now.year - 10, now.month, now.day)
            end = now

            df = yf.download(ticker, start, end)

            if df.empty:
                return Response(
                    {"error": "No data found for the given ticker"},
                    status=status.HTTP_404_NOT_FOUND
                )

            df = df.reset_index()

            # -------------------------------
            # 2️⃣ Basic closing price plot
            # -------------------------------
            plt.switch_backend('AGG')
            plt.figure(figsize=(12, 5))
            plt.plot(df.Close, label='Closing Price')
            plt.title(f'Closing price of {ticker}')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()

            plot_img_path = f'{ticker}_plot.png'
            plot_img = save_plot(plot_img_path)

            # -------------------------------
            # 3️⃣ Moving averages
            # -------------------------------
            ma100 = df.Close.rolling(100).mean()
            plt.figure(figsize=(12, 5))
            plt.plot(df.Close, label='Closing Price')
            plt.plot(ma100, 'r', label='100 DMA')
            plt.legend()
            plot_100_dma = save_plot(f'{ticker}_100_dma.png')

            ma200 = df.Close.rolling(200).mean()
            plt.figure(figsize=(12, 5))
            plt.plot(df.Close, label='Closing Price')
            plt.plot(ma100, 'r', label='100 DMA')
            plt.plot(ma200, 'g', label='200 DMA')
            plt.legend()
            plot_200_dma = save_plot(f'{ticker}_200_dma.png')

            # -------------------------------
            # 4️⃣ Train / Test split
            # -------------------------------
            data_training = pd.DataFrame(df.Close[:int(len(df) * 0.7)])
            data_testing = pd.DataFrame(df.Close[int(len(df) * 0.7):])

            scaler = MinMaxScaler(feature_range=(0, 1))

            # -------------------------------
            # 5️⃣ Load ML model (CRITICAL FIX)
            # -------------------------------
            model_path = os.path.join(settings.BASE_DIR, 'stock_prediction_model.keras')

            if not os.path.exists(model_path):
                return Response(
                    {"error": "Model file not found on server"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            model = load_model(model_path)

            # -------------------------------
            # 6️⃣ Prepare test data
            # -------------------------------
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

            input_data = scaler.fit_transform(final_df)

            x_test, y_test = [], []
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])

            x_test, y_test = np.array(x_test), np.array(y_test)

            # -------------------------------
            # 7️⃣ Predict
            # -------------------------------
            y_predicted = model.predict(x_test)

            y_predicted = scaler.inverse_transform(
                y_predicted.reshape(-1, 1)
            ).flatten()

            y_test = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            ).flatten()

            # -------------------------------
            # 8️⃣ Final prediction plot
            # -------------------------------
            plt.figure(figsize=(12, 5))
            plt.plot(y_test, 'b', label='Original Price')
            plt.plot(y_predicted, 'r', label='Predicted Price')
            plt.legend()

            plot_prediction = save_plot(f'{ticker}_final_prediction.png')

            # -------------------------------
            # 9️⃣ Metrics
            # -------------------------------
            mse = mean_squared_error(y_test, y_predicted)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_predicted)

            return Response(
                {
                    'status': 'success',
                    'plot_img': plot_img,
                    'plot_100_dma': plot_100_dma,
                    'plot_200_dma': plot_200_dma,
                    'plot_prediction': plot_prediction,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2
                },
                status=status.HTTP_200_OK
            )

        except Exception as e:
            import traceback
            print("🔥 STOCK PREDICTION ERROR 🔥")
            print(traceback.format_exc())

            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
