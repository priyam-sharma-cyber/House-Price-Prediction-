# 🏠 House Price Prediction Using Machine Learning

A Machine Learning web application that predicts house prices based on property area and house age using **Linear Regression**. The application is built with Python and Scikit-learn and deployed using Streamlit, providing a simple and interactive interface for users to estimate property prices.

🔗 **Live Demo:** [House Price Prediction App](https://24yawabdgbqnkxpeoga3gu.streamlit.app/)

---

## 📌 Project Overview

House prices depend on several factors, including property size, location, age, and market conditions. This project demonstrates how Machine Learning can be used to estimate house prices based on selected property features.

The application allows users to enter the house area in square feet and the age of the property. A trained Linear Regression model then predicts the estimated price in lakh Indian rupees.

This project is designed as a beginner-friendly Machine Learning project to demonstrate data preparation, model training, prediction, and web application deployment.

## 🎯 Project Objectives

- Build a Machine Learning model to predict house prices.
- Understand and implement Linear Regression.
- Use property area and house age as input features.
- Process structured data using Pandas.
- Train a prediction model using Scikit-learn.
- Build an interactive web application with Streamlit.
- Deploy the application online so users can access it through a browser.

## ✨ Features

- 🏠 **House Price Prediction:** Estimate a property's price from user inputs.
- 📏 **Area-Based Input:** Enter the property area in square feet.
- 📅 **Age-Based Input:** Specify the age of the house in years.
- 🤖 **Machine Learning:** Uses a Linear Regression model for predictions.
- ⚡ **Interactive Interface:** Simple input fields and a prediction button.
- 💰 **Price Output:** Displays the estimated price in lakh INR.
- 🌐 **Live Deployment:** Access the application online using Streamlit.
- 🧹 **Data Preparation:** Converts input columns to numeric values and removes invalid rows before training.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and preprocessing |
| Scikit-learn | Machine Learning model training and prediction |
| Linear Regression | Predicting house prices |
| Streamlit | Building the web application |
| Git and GitHub | Version control and project hosting |
| Streamlit Community Cloud | Deploying the web application |

## 📊 Dataset Information

The application uses a CSV dataset named `house_price_prediction_dataset.csv`.

The model requires the following columns:

| Column | Description |
|---|---|
| `Area_sqft` | Area of the property in square feet |
| `Age_years` | Age of the house in years |
| `Price_Lakh_INR` | House price in lakh Indian rupees; the target variable |

The dataset is used to train the model. The first two columns are the input features, and `Price_Lakh_INR` is the value the model learns to predict.

**Note:** Prediction quality depends on the size, accuracy, and representativeness of the dataset. The model does not automatically account for factors such as location, number of bedrooms, construction quality, or current market trends unless those features are included in the dataset and model.

## ⚙️ How It Works

1. **Load the dataset:** Read the house price data from a CSV file using Pandas.
2. **Prepare the data:** Convert the required columns to numeric values and remove rows containing invalid or missing values.
3. **Select features:** Use `Area_sqft` and `Age_years` as input features.
4. **Train the model:** Fit a Linear Regression model using the input features and known house prices.
5. **Collect user input:** Ask the user for the property area and house age.
6. **Generate a prediction:** Pass the input values to the trained model.
7. **Display the result:** Show the estimated house price in lakh INR through the Streamlit interface.

## 🧠 Machine Learning Algorithm

This project uses **Linear Regression**, a supervised Machine Learning algorithm commonly used to predict continuous numerical values.

The model learns the relationship between the input features and the target price from the training dataset.

A simplified representation of the model is:

`Predicted Price = b0 + b1 × Area_sqft + b2 × Age_years`

Where:

- `b0` is the intercept learned by the model.
- `b1` is the coefficient for property area.
- `b2` is the coefficient for house age.

The model learns these coefficients from the supplied dataset. The formula is a simplified explanation of the prediction process.

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── app.py
├── house_price_prediction_dataset.csv
├── requirements.txt
└── README.md
```

- `app.py` — Contains the Streamlit application and model training logic.
- `house_price_prediction_dataset.csv` — Dataset used for training the model.
- `requirements.txt` — Python dependencies required by the application.
- `README.md` — Project documentation.

## 🚀 Run the Project Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Open the project folder

```bash
cd House-Price-Prediction
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

### 5. Open the application

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open this address in your browser to use the application.

## 📦 Requirements

Add the following packages to `requirements.txt`:

```text
streamlit
pandas
scikit-learn
```

These dependencies provide the web interface, data processing, and Machine Learning functionality.

## 🌐 Live Application

Try the deployed application here:

**[Open House Price Prediction App](https://24yawabdgbqnkxpeoga3gu.streamlit.app/)**

Enter the house area and age, click the prediction button, and view the estimated property price.

## 🔮 Future Improvements

Potential improvements for future versions include:

- Add more features such as location, number of bedrooms, bathrooms, and property type.
- Use
