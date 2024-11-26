import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
from datetime import datetime, time

# Multiplying factors for each stock
multiplying_factors = {
    "GOOGL": 1.6,
    "AAPL": 1.55,
    "MSFT": 1.71,
    "META": 1.3,
    "AMZN": 1.7,
    "NFLX": 1.35,
    "NVDA": 1.2
}

# Market hours (9:30 AM to 4:00 PM)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Load scaler and model dynamically
@st.cache_resource
def load_model_and_scaler(stock_name):
    model_path = f"models/{stock_name.lower()}_lstm.h5"
    model = tf.keras.models.load_model(model_path)

    scaler_path = f"scalers/{stock_name.lower()}_scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler


# Load historical data
def load_historical_data(stock_name):
    file_path = f"data/{stock_name.upper()}.csv"
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df[['Date', 'Close']]
    except FileNotFoundError:
        st.error(f"Historical data for {stock_name} not found.")
        return pd.DataFrame()


# Preprocess and reverse scale functions
def preprocess_input(scaler, data):
    return scaler.transform(data)


def reverse_scaling(scaler, data):
    return data / scaler.scale_[0]


# Generate predictions for future or missing past dates
def generate_future_predictions(model, scaler, last_100_days, num_days):
    future_predictions = []
    scaled_input = preprocess_input(scaler, last_100_days)

    for _ in range(num_days):
        scaled_input_batch = np.expand_dims(scaled_input, axis=0)
        prediction_scaled = model.predict(scaled_input_batch)[0][0]
        prediction = reverse_scaling(scaler, prediction_scaled)
        future_predictions.append(prediction)
        next_scaled_input = np.array([[prediction_scaled]])
        scaled_input = np.append(scaled_input, next_scaled_input, axis=0)[1:]

    return future_predictions


# Streamlit App Configuration
st.set_page_config(page_title="Stock Price Prediction", page_icon="📊", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            color: #0056b3;
            font-weight: bold;
        }
        .section-title {
            color: #0056b3;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
        .prediction-card {
            background-color: #0056b3;
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            font-size: 20px;
        }
        .stButton>button {
            background-color: #0056b3;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #004080;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<div class="main-title">Stock Price Prediction App</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Inputs")
stock_options = ["GOOGL", "AAPL", "MSFT", "META", "AMZN", "NFLX", "NVDA"]
stock_name = st.sidebar.selectbox("Select a Stock", stock_options)
selected_date = st.sidebar.date_input("Select a Date", min_value=datetime(2004, 1, 1))

# Load data and models
model, scaler = load_model_and_scaler(stock_name)
historical_data = load_historical_data(stock_name)

# Ensure historical data is available
if not historical_data.empty:
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

    # Get today's date and time
    today = datetime.now().date()
    current_time = datetime.now().time()

    if selected_date < today:
        # Handle past dates
        if selected_date not in historical_data['Date'].dt.date.values:
            # Predict for a missing past date
            historical_data = historical_data.tail(100)
            last_100_days = historical_data['Close'].values.reshape(-1, 1)
            predicted_past_price = generate_future_predictions(model, scaler, last_100_days, 1)[0]
            adjusted_past_price = predicted_past_price * multiplying_factors[stock_name]
            st.markdown(f"""
                <div class="prediction-card">
                    <strong>Predicted Price for {stock_name} on {selected_date}: </strong> 
                    ${adjusted_past_price:.2f}
                </div>
            """, unsafe_allow_html=True)
        else:
            # Show actual historical price
            historical_price = historical_data.loc[historical_data['Date'] == pd.to_datetime(selected_date), 'Close']
            st.markdown(f"""
                <div class="prediction-card">
                    <strong>Actual Price for {stock_name} on {selected_date}: </strong> 
                    ${historical_price.values[0]:.2f}
                </div>
            """, unsafe_allow_html=True)

    elif selected_date == today:
        # Handle today's date
        if MARKET_OPEN <= current_time <= MARKET_CLOSE:
            # Predict today's price (market open)
            historical_data = historical_data.tail(100)
            last_100_days = historical_data['Close'].values.reshape(-1, 1)
            predicted_today_price = generate_future_predictions(model, scaler, last_100_days, 1)[0]
            adjusted_today_price = predicted_today_price * multiplying_factors[stock_name]
            st.markdown(f"""
                <div class="prediction-card">
                    <strong>Predicted Price for {stock_name} (Market Open): </strong> 
                    ${adjusted_today_price:.2f}
                </div>
            """, unsafe_allow_html=True)
        else:
            # Show today's actual price (market closed)
            historical_price = historical_data.iloc[-1]['Close']
            st.markdown(f"""
                <div class="prediction-card">
                    <strong>Actual Price for {stock_name} Today (Market Closed): </strong> 
                    ${historical_price:.2f}
                </div>
            """, unsafe_allow_html=True)

    else:
        # Predict for future dates
        num_days = (selected_date - today).days
        historical_data = historical_data.tail(100)
        last_100_days = historical_data['Close'].values.reshape(-1, 1)
        future_predictions = generate_future_predictions(model, scaler, last_100_days, num_days)
        predicted_future_price = future_predictions[-1] * multiplying_factors[stock_name]
        st.markdown(f"""
            <div class="prediction-card">
                <strong>Predicted Price for {stock_name} on {selected_date}: </strong> 
                ${predicted_future_price:.2f}
            </div>
        """, unsafe_allow_html=True)
else:
    st.error("Historical data unavailable for the selected stock.")
