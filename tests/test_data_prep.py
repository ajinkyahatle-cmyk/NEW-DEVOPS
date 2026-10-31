import pandas as pd
import pytest

from src.data_prep import (
    RAW_FEATURE_COLUMNS,
    prepare_features,
)


def create_sample_data():

    sample = {
        "date": ["20141013T000000"],
    }

    for column in RAW_FEATURE_COLUMNS:
        sample[column] = [1]

    sample["bedrooms"] = [3]
    sample["bathrooms"] = [2.0]
    sample["sqft_living"] = [1800]
    sample["sqft_lot"] = [5000]
    sample["floors"] = [1.0]
    sample["zipcode"] = [98101]

    return pd.DataFrame(sample)


def test_prepare_features_returns_expected_columns():

    data = create_sample_data()

    result = prepare_features(data)

    assert len(result.columns) == 21

    assert "sale_year" in result.columns
    assert "sale_month" in result.columns
    assert "sale_day" in result.columns

    assert "date" not in result.columns


def test_date_features_are_created():

    data = create_sample_data()

    result = prepare_features(data)

    assert result.iloc[0]["sale_year"] == 2014
    assert result.iloc[0]["sale_month"] == 10
    assert result.iloc[0]["sale_day"] == 13


def test_missing_column_raises_error():

    data = create_sample_data()

    data = data.drop(columns=["bedrooms"])

    with pytest.raises(ValueError):
        prepare_features(data)


def test_invalid_date_raises_error():

    data = create_sample_data()

    data.loc[0, "date"] = "not-a-date"

    with pytest.raises(ValueError):
        prepare_features(data)