from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from .serializers import StockPredictionSerializer
from .ml_client import get_stock_prediction


class StockPredictionAPIView(APIView):
    permission_classes = (AllowAny,)

    def post(self, request):
        serializer = StockPredictionSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )

        ticker = serializer.validated_data["ticker"]

        ml_result = get_stock_prediction(ticker)

        if "error" in ml_result:
            return Response(
                ml_result,
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        return Response(
            {
                "status": "success",
                "source": "fastapi-ml-service",
                "data": ml_result
            },
            status=status.HTTP_200_OK
        )
