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


# Generate iterative predictions for future dates
def generate_future_predictions(model, scaler, last_100_days, num_days):
    future_predictions = []

    # Scale the initial input data
    scaled_input = preprocess_input(scaler, last_100_days)

    for _ in range(num_days):
        # Reshape input for prediction
        scaled_input_batch = np.expand_dims(scaled_input, axis=0)

        # Predict the next value
        prediction_scaled = model.predict(scaled_input_batch)[0][0]

        # Reverse scaling to get the actual predicted value
        prediction = reverse_scaling(scaler, prediction_scaled)
        future_predictions.append(prediction)

        # Update the input data by appending the prediction
        next_scaled_input = np.array([[prediction_scaled]])
        scaled_input = np.append(scaled_input, next_scaled_input, axis=0)[1:]  # Keep only the last 100 entries

    return future_predictions


# Streamlit app
st.title("Stock Price Prediction App")
st.write("Predict stock prices for GOOGL, AAPL, MSFT, META, AMZN, NFLX, and NVDA.")

# Sidebar inputs
stock_options = ["GOOGL", "AAPL", "MSFT", "META", "AMZN", "NFLX", "NVDA"]
stock_name = st.sidebar.selectbox("Select Stock", stock_options)
date_input = st.sidebar.date_input(
    "Select Date (Historical or Future)",
    min_value=datetime(2004, 1, 1),
    max_value=datetime(2024, 12, 31),
)

# Predict button
if st.sidebar.button("Predict"):
    # Load the model and scaler
    model, scaler = load_model_and_scaler(stock_name)

    # Load historical data
    historical_data = load_historical_data(stock_name)

    if historical_data.empty:
        st.error("Unable to load historical data for the selected stock.")
    else:
        # Use the last 100 days of data for prediction
        historical_data = historical_data.tail(100)  # Use only the last 100 days
        last_100_days = historical_data['Close'].values.reshape(-1, 1)

        # Determine the number of days to predict
        today = datetime.now().date()
        num_days = max((date_input - today).days, 0)  # Future date difference in days

        if num_days == 0:
            st.error("Please select a future date to predict.")
        else:
            # Generate future predictions
            future_predictions = generate_future_predictions(model, scaler, last_100_days, num_days)

            # Get the prediction for the selected future date
            predicted_price = future_predictions[-1]

            # Display the result
            st.write(f"**Predicted Price for {stock_name} on {date_input}:** ${predicted_price:.2f}")
