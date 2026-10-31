from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
import plotly.express as px

from src.predict import predict_house_price


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EstateIQ | House Price Prediction",
    page_icon="🏡",
    layout="wide",
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "kc_house_data.csv"


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #f5f5f5;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 16px;
    }

    .price-card {
        background: #1c2533;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #334155;
    }

    div.stButton > button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        height: 48px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD DATASET FOR DASHBOARD VISUALIZATIONS
# --------------------------------------------------

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        return None

    return pd.read_csv(DATA_PATH)


data = load_dataset()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏡 EstateIQ</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'House Price Prediction powered by PyCaret and DevOps'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("EstateIQ")
st.sidebar.caption("Machine Learning Prediction System")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Predict House Price",
    ],
)


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    st.subheader("Dataset Overview")

    if data is None:

        st.warning(
            "Dataset not found. Check data/raw/kc_house_data.csv."
        )

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            f"{len(data):,}"
        )

        col2.metric(
            "Average House Price",
            f"${data['price'].mean():,.0f}"
        )

        col3.metric(
            "Median House Price",
            f"${data['price'].median():,.0f}"
        )

        st.divider()

        st.subheader("Historical House Price Distribution")

        fig = px.histogram(
            data,
            x="price",
            nbins=50,
            title="Distribution of Historical House Prices",
            labels={
                "price": "House Price (USD)"
            },
        )

        fig.update_layout(
            template="plotly_dark",
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

elif page == "Predict House Price":

    st.subheader("Enter House Details")

    st.write(
        "Provide the property details below to generate "
        "a house-price prediction."
    )

    with st.form("house_prediction_form"):

        st.markdown("### Property Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            bedrooms = st.number_input(
                "Bedrooms",
                min_value=0,
                max_value=20,
                value=3,
            )

            bathrooms = st.number_input(
                "Bathrooms",
                min_value=0.0,
                max_value=15.0,
                value=2.0,
                step=0.25,
            )

            floors = st.number_input(
                "Floors",
                min_value=1.0,
                max_value=5.0,
                value=1.0,
                step=0.5,
            )

            waterfront = st.selectbox(
                "Waterfront",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

            view = st.selectbox(
                "View Rating",
                [0, 1, 2, 3, 4],
            )

            condition = st.selectbox(
                "Condition",
                [1, 2, 3, 4, 5],
                index=2,
            )

        with c2:
            sqft_living = st.number_input(
                "Living Area (sqft)",
                min_value=100,
                max_value=20000,
                value=1800,
            )

            sqft_lot = st.number_input(
                "Lot Area (sqft)",
                min_value=100,
                max_value=1000000,
                value=5000,
            )

            grade = st.number_input(
                "Construction Grade",
                min_value=1,
                max_value=13,
                value=7,
            )

            sqft_above = st.number_input(
                "Above Ground Area (sqft)",
                min_value=0,
                max_value=20000,
                value=1500,
            )

            sqft_basement = st.number_input(
                "Basement Area (sqft)",
                min_value=0,
                max_value=10000,
                value=300,
            )

            zipcode = st.number_input(
                "Zipcode",
                min_value=98000,
                max_value=99000,
                value=98101,
            )

        with c3:
            yr_built = st.number_input(
                "Year Built",
                min_value=1800,
                max_value=2026,
                value=1990,
            )

            yr_renovated = st.number_input(
                "Year Renovated (0 if never)",
                min_value=0,
                max_value=2026,
                value=0,
            )

            lat = st.number_input(
                "Latitude",
                min_value=47.0,
                max_value=48.0,
                value=47.50,
                format="%.6f",
            )

            long = st.number_input(
                "Longitude",
                min_value=-123.0,
                max_value=-121.0,
                value=-122.20,
                format="%.6f",
            )

            sqft_living15 = st.number_input(
                "Nearby Average Living Area (sqft)",
                min_value=100,
                max_value=20000,
                value=1800,
            )

            sqft_lot15 = st.number_input(
                "Nearby Average Lot Area (sqft)",
                min_value=100,
                max_value=1000000,
                value=5000,
            )

        st.markdown("### Sale Information")

        sale_date = st.date_input(
            "Sale Date",
            value=date(2015, 1, 1),
        )

        submitted = st.form_submit_button(
            "Predict House Price"
        )

    # --------------------------------------------------
    # GENERATE PREDICTION
    # --------------------------------------------------

    if submitted:

        house_data = pd.DataFrame(
            [
                {
                    "date": sale_date.strftime("%Y%m%dT000000"),
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "sqft_living": sqft_living,
                    "sqft_lot": sqft_lot,
                    "floors": floors,
                    "waterfront": waterfront,
                    "view": view,
                    "condition": condition,
                    "grade": grade,
                    "sqft_above": sqft_above,
                    "sqft_basement": sqft_basement,
                    "yr_built": yr_built,
                    "yr_renovated": yr_renovated,
                    "zipcode": zipcode,
                    "lat": lat,
                    "long": long,
                    "sqft_living15": sqft_living15,
                    "sqft_lot15": sqft_lot15,
                }
            ]
        )

        try:

            with st.spinner("Generating prediction..."):

                predicted_price = predict_house_price(
                    house_data
                )

            st.success("Prediction generated successfully!")

            st.markdown(
                f"""
                <div class="price-card">
                    <h3>Estimated House Price</h3>
                    <h1>${predicted_price:,.2f}</h1>
                    <p>Generated using the trained PyCaret pipeline</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "This is a model-generated estimate based on "
                "historical King County sales data, not a live "
                "property valuation."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )