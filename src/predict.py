from functools import lru_cache
from pathlib import Path
from urllib.request import Request, urlopen
import shutil

import pandas as pd
from pycaret.regression import load_model, predict_model

from src.data_prep import prepare_features


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline"
MODEL_FILE = Path(str(MODEL_PATH) + ".pkl")

MODEL_URL = (
    "https://github.com/ajinkyahatle-cmyk/NEW-DEVOPS/"
    "releases/download/v1.0.0/house_price_pipeline.pkl"
)


def download_model():
    """Download the trained model if it is missing."""

    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 0:
        return

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = MODEL_FILE.with_suffix(".pkl.tmp")

    print("EstateIQ: Downloading trained model...")

    try:
        request = Request(
            MODEL_URL,
            headers={"User-Agent": "EstateIQ-Streamlit-App"}
        )

        with urlopen(request, timeout=120) as response:
            with open(temporary_file, "wb") as output:
                shutil.copyfileobj(response, output)

        if not temporary_file.exists() or temporary_file.stat().st_size == 0:
            raise RuntimeError("Downloaded model file is empty.")

        temporary_file.replace(MODEL_FILE)

        print("EstateIQ: Model downloaded successfully.")

    except Exception as error:
        if temporary_file.exists():
            temporary_file.unlink()

        raise RuntimeError(
            "Could not download the EstateIQ model. "
            "Check that the GitHub Release is public and the download URL is correct. "
            f"Details: {error}"
        ) from error


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the trained PyCaret model."""

    download_model()

    return load_model(str(MODEL_PATH))


def predict_house_price(house_data: pd.DataFrame) -> float:
    """Generate a house-price prediction."""

    model = get_model()

    prepared_data = prepare_features(house_data)

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