import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Helper function to load the scaler
def load_scaler(stock_name):
    scaler_path = f"scalers/{stock_name.lower()}_scaler.pkl"
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if not isinstance(scaler, MinMaxScaler):
            raise ValueError(f"Scaler object for {stock_name} is invalid. Check the scaler file.")
        return scaler
    except FileNotFoundError:
        st.error(f"Scaler file not found for {stock_name}. Please ensure the scaler is saved at {scaler_path}.")
        return None

# Helper function to preprocess the input date
def preprocess_input(scaler, data):
    data_reshaped = np.array(data).reshape(-1, 1)  # Ensure data is in 2D array
    return scaler.transform(data_reshaped)

# Helper function to load the stock model
def load_stock_model(stock_name):
    model_path = f"models/{stock_name.lower()}_lstm.h5"
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model for {stock_name}: {str(e)}")
        return None

# Streamlit App
st.title("Stock Price Prediction App")
st.write("Predict stock prices for GOOGL, AAPL, MSFT, META, AMZN, NFLX, and NVDA.")

# Sidebar inputs
stock_name = st.sidebar.selectbox("Select Stock", ["GOOGL", "AAPL", "MSFT", "META", "AMZN", "NFLX", "NVDA"])
date_input = st.sidebar.date_input("Select Date", min_value=datetime(2004, 1, 1), max_value=datetime(2024, 11, 22))

# Main logic
if st.sidebar.button("Predict"):
    # Load the scaler and model
    scaler = load_scaler(stock_name)
    model = load_stock_model(stock_name)

    if scaler is not None and model is not None:
        # Preprocess the input
        date_numeric = (datetime.combine(date_input, datetime.min.time()) - datetime(1970, 1, 1)).days  # Convert date to numeric
        scaled_date = preprocess_input(scaler, [date_numeric])

        # Make prediction
        scaled_date = np.expand_dims(scaled_date, axis=0)  # Reshape for model input
        prediction_scaled = model.predict(scaled_date)
        
        # Reverse scaling
        scale_factor = 1 / scaler.scale_[0]  # Get the scaling factor
        prediction = prediction_scaled[0][0] * scale_factor

        # Display result
        st.write(f"**Predicted Price for {stock_name} on {date_input}:** ${prediction:.2f}")

        # Optional: If the date is historical, show actual price
        if date_input <= datetime(2024, 11, 22).date():  # Ensure date comparison works
            try:
                historical_data = pd.read_csv(f"data/{stock_name.lower()}.csv")
                st.write("Columns in the historical data:", historical_data.columns)

                historical_data['Date'] = pd.to_datetime(historical_data['Date']).dt.date  # Ensure comparison works
                actual_price = historical_data.loc[historical_data['Date'] == date_input, 'Close']
                if not actual_price.empty:
                    st.write(f"**Actual Price for {stock_name} on {date_input}:** ${actual_price.values[0]:.2f}")
                else:
                    st.write("Actual price data not available for the selected date.")
            except FileNotFoundError:
                st.error(f"Historical data file not found for {stock_name}. Ensure the file exists.")
    else:
        st.error("Failed to load model or scaler. Please check your files.")
