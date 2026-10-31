from pathlib import Path

from pycaret.regression import (
    setup,
    create_model,
    finalize_model,
    save_model,
)

from src.data_prep import load_training_data


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "raw" / "kc_house_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_NAME = "house_price_pipeline"


def train_pipeline():

    print("\nStarting PyCaret training pipeline...")

    # Check dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Create model directory
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load and prepare dataset
    X, y = load_training_data(str(DATA_PATH))

    # Combine features and target for PyCaret
    training_data = X.copy()
    training_data["price"] = y

    print(f"Training rows: {len(training_data)}")
    print(f"Number of input features: {len(X.columns)}")

    # Initialize PyCaret experiment
    setup(
        data=training_data,
        target="price",
        session_id=42,
        normalize=True,
        numeric_imputation="median",
        html=False,
        verbose=False,
        n_jobs=-1,
    )

    print("\nTraining Extra Trees Regressor...")

    # Train the selected model
    model = create_model(
        "et",
        cross_validation=False,
        verbose=False,
    )

    # Finalize model using the complete dataset
    final_model = finalize_model(model)

    # Save trained pipeline
    model_path = MODEL_DIR / MODEL_NAME

    save_model(
        final_model,
        str(model_path)
    )

    print("\nTraining completed successfully!")
    print(f"Saved model: {model_path}.pkl")


if __name__ == "__main__":
    train_pipeline()