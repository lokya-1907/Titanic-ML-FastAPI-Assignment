"""Simple Python client for the FastAPI /predict endpoint."""
import requests

API_URL = "http://127.0.0.1:8000"

def main():
    passenger = {
        "Pclass": 3,
        "Sex": "female",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    response = requests.post(f"{API_URL}/predict", json=passenger, timeout=10)
    print(f"HTTP status: {response.status_code}")
    response.raise_for_status()
    print("API prediction response:")
    print(response.json())

if __name__ == "__main__":
    main()
