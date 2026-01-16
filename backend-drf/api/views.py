from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import requests

from .serializers import StockPredictionSerializer


# 🔗 STREAMLIT ML SERVICE URL
ML_SERVICE_URL = "https://stock-prediction-app-wrxuhbc7wx2qmcbcbybh6d.streamlit.app"


class StockPredictionAPIView(APIView):
    permission_classes = (AllowAny,)

    def post(self, request):
        # 1️⃣ Validate request
        serializer = StockPredictionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )

        ticker = serializer.validated_data["ticker"]

        # 2️⃣ Call ML service
        try:
            ml_response = requests.get(
                ML_SERVICE_URL,
                params={"ticker": ticker},
                timeout=30
            )

            # If ML service returns error
            if ml_response.status_code != 200:
                return Response(
                    {
                        "error": "ML service error",
                        "details": ml_response.text
                    },
                    status=ml_response.status_code
                )

            # 3️⃣ Return ML response directly
            return Response(
                ml_response.json(),
                status=status.HTTP_200_OK
            )

        except requests.exceptions.Timeout:
            return Response(
                {"error": "ML service timeout"},
                status=status.HTTP_504_GATEWAY_TIMEOUT
            )

        except requests.exceptions.ConnectionError:
            return Response(
                {"error": "ML service unreachable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
