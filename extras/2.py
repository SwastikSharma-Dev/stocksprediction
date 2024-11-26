import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime


# Load scaler and model dynamically
@st.cache_resource
def load_model_and_scaler(stock_name):
    # Load model
    model_path = f"models/{stock_name.lower()}_lstm.h5"
    model = tf.keras.models.load_model(model_path)

    # Load scaler
    scaler_path = f"scalers/{stock_name.lower()}_scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


# Load historical data from CSV
def load_historical_data(stock_name):
    file_path = f"data/{stock_name.upper()}.csv"  # Ensure file path matches the naming convention
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
        return df[['Date', 'Close']]  # Return only relevant columns
    except FileNotFoundError:
        st.error(f"File not found for {stock_name}. Please ensure the file exists in the `data/` folder.")
        return pd.DataFrame()  # Return empty DataFrame if file is missing


# Preprocess input data using the scaler
def preprocess_input(scaler, data):
    scaled_data = scaler.transform(data)
    return scaled_data


# Postprocess predictions to reverse the scaling
def reverse_scaling(scaler, data):
    scale_factor = 1 / scaler.scale_[0]  # Assuming 'Close' is the first feature
    return data * scale_factor


# Streamlit app
st.title("Stock Price Prediction App")
st.write("Predict stock prices for GOOGL, AAPL, MSFT, META, AMZN, NFLX, and NVDA.")

# Sidebar inputs
stock_options = ["GOOGL", "AAPL", "MSFT", "META", "AMZN", "NFLX", "NVDA"]
stock_name = st.sidebar.selectbox("Select Stock", stock_options)
date_input = st.sidebar.date_input("Select Date", min_value=datetime(2004, 1, 1), max_value=datetime(2024, 11, 22))

# Predict button
if st.sidebar.button("Predict"):
    # Load the model and scaler
    model, scaler = load_model_and_scaler(stock_name)

    # Load historical data
    historical_data = load_historical_data(stock_name)

    if historical_data.empty:
        st.error("Unable to load historical data for the selected stock.")
    else:
        # Ensure the selected date is within the historical data range
        if date_input > historical_data['Date'].max().date():
            st.error("Date is beyond the available historical data. Please choose a date within the range.")
        else:
            # Get the last 100 days of data before the selected date
            historical_data = historical_data[historical_data['Date'] < pd.Timestamp(date_input)]
            if len(historical_data) < 100:
                st.error("Not enough data available for the last 100 days to make a prediction.")
            else:
                last_100_days = historical_data.tail(100)['Close'].values.reshape(-1, 1)

                # Scale the data
                scaled_input = preprocess_input(scaler, last_100_days)
                scaled_input = np.expand_dims(scaled_input, axis=0)  # Add batch dimension

                # Predict
                prediction_scaled = model.predict(scaled_input)

                # Reverse scaling
                prediction = reverse_scaling(scaler, prediction_scaled[0][0])

                # Display results
                st.write(f"**Predicted Price for {stock_name} on {date_input}:** ${prediction:.2f}")

                # Show the latest actual closing price
                latest_close = historical_data['Close'].iloc[-1]  # Extract scalar value
                st.write(f"**Latest Actual Closing Price for {stock_name}:** ${latest_close:.2f}")
