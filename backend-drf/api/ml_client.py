import requests

ML_API_URL = "https://stock-prediction-portal-production-79ec.up.railway.app/predict"

def get_stock_prediction(ticker: str):
    try:
        response = requests.get(
            ML_API_URL,
            params={"ticker": ticker},
            timeout=15
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        return {
            "error": "ML service unavailable",
            "details": str(e)
        }
