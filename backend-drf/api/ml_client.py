import requests
import os

ML_API_BASE_URL = os.getenv("https://stock-prediction-portal-production-79ec.up.railway.app")

def get_stock_prediction(ticker: str):
    try:
        if not ML_API_BASE_URL:
            return {
                "error": "ML_API_BASE_URL not configured"
            }

        response = requests.get(
            f"{ML_API_BASE_URL}/predict",
            params={"ticker": ticker},
            timeout=5
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {
            "error": "ML service unavailable",
            "details": str(e)
        }
