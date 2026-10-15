
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Predictor")
st.caption("Estimate your property's price in seconds.")

@st.cache_resource
def train_model():
    df = pd.read_csv("house_price_prediction_dataset.csv")

    features = ["Area_sqft", "Age_years"]
    target = "Price_Lakh_INR"

    df = df[features + [target]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(df) < 2:
        raise ValueError("Dataset mein kam se kam 2 valid rows honi chahiye.")

    model = LinearRegression()
    model.fit(df[features], df[target])

    return model

try:
    model = train_model()

    st.subheader("Enter Property Details")

    area = st.number_input(
        "House Area (sq. ft.)",
        min_value=100,
        max_value=100000,
        value=1000,
        step=100
    )

    age = st.number_input(
        "House Age (years)",
        min_value=0,
        max_value=200,
        value=5,
        step=1
    )

    if st.button("Predict House Price", use_container_width=True):
        prediction = model.predict(
            pd.DataFrame({
                "Area_sqft": [area],
                "Age_years": [age]
            })
        )[0]

        if prediction >= 0:
            st.success(f"Estimated House Price: ₹{prediction:.2f} Lakh")
        else:
            st.warning(
                f"Model ne negative estimate diya: ₹{prediction:.2f} Lakh. "
                "Dataset aur model ko check karein."
            )

except FileNotFoundError:
    st.error("Dataset file nahi mili. CSV ko app.py ke saath same folder mein rakho.")
except Exception as e:
    st.error(f"App error: {e}")
