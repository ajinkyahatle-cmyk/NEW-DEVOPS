import pandas as pd


RAW_FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

REQUIRED_COLUMNS = ["date"] + RAW_FEATURE_COLUMNS


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw house data into the exact feature format
    expected by the trained PyCaret model.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = data[REQUIRED_COLUMNS].copy()

    # Convert sale date into datetime
    sale_dates = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if sale_dates.isna().any():
        raise ValueError(
            "One or more sale dates are invalid."
        )

    # Extract date-based features
    df["sale_year"] = sale_dates.dt.year
    df["sale_month"] = sale_dates.dt.month
    df["sale_day"] = sale_dates.dt.day

    # Remove original date column
    df = df.drop(columns=["date"])

    # Convert every feature to numeric
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Keep a consistent feature order
    feature_columns = RAW_FEATURE_COLUMNS + [
        "sale_year",
        "sale_month",
        "sale_day",
    ]

    return df[feature_columns]


def load_training_data(csv_path: str):
    """
    Load the CSV and separate the input features
    from the target variable.
    """

    data = pd.read_csv(csv_path)

    if "price" not in data.columns:
        raise ValueError(
            "The dataset must contain a 'price' column."
        )

    # Convert target to numeric
    data["price"] = pd.to_numeric(
        data["price"],
        errors="coerce"
    )

    # Remove rows without a valid target
    data = data.dropna(subset=["price"]).copy()

    # Prepare model features
    X = prepare_features(data)

    # Target variable
    y = data["price"].reset_index(drop=True)

    return X.reset_index(drop=True), y