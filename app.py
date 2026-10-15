

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write(
    "Predict house prices using Machine Learning "
    "and Linear Regression."
)

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your house price CSV file",
    type=["csv"]
)

# Load the dataset
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif Path("house_price_prediction_dataset.csv").exists():
        df = pd.read_csv("house_price_prediction_dataset.csv")
    else:
        st.info(
            "Please upload your house_price_prediction_dataset.csv "
            "file using the sidebar."
        )
        st.stop()

    required_columns = [
        "Area_sqft",
        "Age_years",
        "Price_Lakh_INR"
    ]

    if not all(col in df.columns for col in required_columns):
        st.error(
            "Dataset must contain: Area_sqft, Age_years, "
            "Price_Lakh_INR"
        )
        st.stop()

    df = df[required_columns].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(df) < 5:
        st.error("Dataset must contain at least 5 valid rows.")
        st.stop()

    if (df["Area_sqft"] < 0).any() or (df["Age_years"] < 0).any():
        st.error("Area and age cannot be negative.")
        st.stop()

    if not np.isfinite(df.to_numpy()).all():
        st.error("Dataset contains invalid numeric values.")
        st.stop()

    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Houses", len(df))
    col2.metric("Average Price", f"₹{df['Price_Lakh_INR'].mean():.2f} Lakh")
    col3.metric("Average Area", f"{df['Area_sqft'].mean():.0f} sq ft")

    # Prepare features and target
    X = df[["Age_years", "Area_sqft"]]
    y = df["Price_Lakh_INR"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    st.subheader("📈 Model Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{mean_absolute_error(y_test, predictions):.3f}")
    c2.metric("MSE", f"{mse:.3f}")
    c3.metric("RMSE", f"{np.sqrt(mse):.3f}")
    c4.metric("R² Score", f"{r2_score(y_test, predictions):.3f}")

    # Prediction section
    st.subheader("🏡 Predict Your House Price")

    area = st.number_input(
        "House Area (square feet)",
        min_value=1,
        value=1000,
        step=100
    )

    age = st.number_input(
        "House Age (years)",
        min_value=0,
        value=10,
        step=1
    )

    if st.button("Predict House Price", type="primary"):
        new_house = pd.DataFrame({
            "Age_years": [age],
            "Area_sqft": [area]
        })

        predicted_price = model.predict(new_house)[0]

        st.success(
            f"Estimated House Price: ₹{predicted_price:.2f} Lakh"
        )

    st.caption(
        "Note: Predictions are estimates based on the uploaded dataset "
        "and are not guaranteed market prices."
    )

except (ValueError, pd.errors.ParserError) as e:
    st.error(f"Unable to process the dataset: {e}")
