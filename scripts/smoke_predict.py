"""Smoke test: load the saved PyCaret pipeline and make one prediction.

Run from the project root, on the host or inside the Docker image:
    python -m scripts.smoke_predict
"""

import pandas as pd

from src.predict import predict_house_price

SAMPLE_HOUSE = {
    "date": "20150101T000000",
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 1800,
    "sqft_lot": 5000,
    "floors": 1.0,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1500,
    "sqft_basement": 300,
    "yr_built": 1990,
    "yr_renovated": 0,
    "zipcode": 98101,
    "lat": 47.50,
    "long": -122.20,
    "sqft_living15": 1800,
    "sqft_lot15": 5000,
}

if __name__ == "__main__":
    price = float(predict_house_price(pd.DataFrame([SAMPLE_HOUSE])))
    assert 0 < price < 10_000_000, f"Implausible prediction: {price}"
    print(f"Smoke prediction OK: ${price:,.0f}")