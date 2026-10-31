# PyCaret ML Pipeline - House Price Prediction

## Project Overview

This project implements a machine learning pipeline using
PyCaret to predict house prices from historical King County
house-sales data.

The application uses Streamlit for its user interface and
Docker for containerization.

GitHub Actions automates testing and Docker image building.

## Technologies

- Python 3.11
- PyCaret
- Pandas
- Streamlit
- Plotly
- Docker
- GitHub Actions
- Pytest

## Project Workflow

1. Load the historical house-sales dataset.
2. Prepare the input features.
3. Train an Extra Trees Regression model using PyCaret.
4. Save the trained model.
5. Load the model for predictions.
6. Display predictions through Streamlit.
7. Run automated tests using GitHub Actions.
8. Build a Docker image.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt