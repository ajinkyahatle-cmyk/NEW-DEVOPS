from functools import lru_cache
from pathlib import Path

import pandas as pd

from pycaret.regression import load_model, predict_model

from src.data_prep import prepare_features


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline"


@lru_cache(maxsize=1)
def get_model():
    """
    Load the saved model once and reuse it.
    """

    model_file = Path(str(MODEL_PATH) + ".pkl")

    if not model_file.exists():
        raise FileNotFoundError(
            "Trained model not found. "
            "Run: python -m src.train"
        )

    return load_model(str(MODEL_PATH))


def predict_house_price(house_data: pd.DataFrame) -> float:
    """
    Predict the price of a new house.
    """

    model = get_model()

    # Use exactly the same preprocessing as training
    prepared_data = prepare_features(house_data)

    # Generate prediction
    predictions = predict_model(
        model,
        data=prepared_data
    )

    if "prediction_label" not in predictions.columns:
        raise RuntimeError(
            "PyCaret did not return prediction_label."
        )

    predicted_price = float(
        predictions["prediction_label"].iloc[0]
    )

    return predicted_price