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
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# Advanced CSS for Cleaner Design
st.markdown("""
    <style>
        /* Enhanced Full-screen layout */
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            font-family: 'Inter', 'Roboto', sans-serif;
        }

        /* Main Page Heading */
        .main-heading {
            font-size: 3.5rem;
            font-weight: 900;
            color: white;
            text-align: center;
            margin-bottom: 40px;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
            letter-spacing: -2px;
        }

        /* Prediction Container */
        .prediction-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.125);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
            padding: 40px;
            text-align: center;
            width: 80%;
            max-width: 700px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        /* Large Prediction Value */
        .prediction-value {
            font-size: 4rem;
            font-weight: 800;
            color: #ffffff;
            margin: 20px 0;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
            letter-spacing: -2px;
        }

        /* Prediction Date Text */
        .prediction-text {
            font-size: 2rem;
            color: rgba(255, 255, 255, 0.7);
            margin-top: 20px;
            text-align: center;
        }

        /* Responsive Sidebar */
        .css-1aumxhk {
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.125);
        }

        /* Buttons and Inputs */
        .stButton>button {
            background-color: rgba(0, 255, 255, 0.2);
            color: white;
            border: 1px solid rgba(0, 255, 255, 0.5);
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: rgba(0, 255, 255, 0.4);
            transform: scale(1.05);
        }

        .stSelectbox, .stDateInput {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
        }

        /* Ensure centered content */
        .stMarkdown-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Main Page Heading
st.markdown('<div class="main-heading">Stock Price Prediction</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 Stock Prediction")
stock_options = ["GOOGL", "AAPL", "MSFT", "META", "AMZN", "NFLX", "NVDA"]
stock_name = st.sidebar.selectbox("Choose a Stock", stock_options)
selected_date = st.sidebar.date_input("Select Prediction Date", min_value=datetime(2004, 1, 1))

# Main Prediction Logic
def get_stock_prediction(stock_name, selected_date):
    # Load data and models
    model, scaler = load_model_and_scaler(stock_name)
    historical_data = load_historical_data(stock_name)

    if historical_data.empty:
        st.error("Historical data unavailable for the selected stock.")
        return None

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
            return predicted_past_price * multiplying_factors[stock_name]
        else:
            # Show actual historical price
            historical_price = historical_data.loc[historical_data['Date'] == pd.to_datetime(selected_date), 'Close']
            return historical_price.values[0]

    elif selected_date == today:
        # Handle today's date
        if MARKET_OPEN <= current_time <= MARKET_CLOSE:
            # Predict today's price (market open)
            historical_data = historical_data.tail(100)
            last_100_days = historical_data['Close'].values.reshape(-1, 1)
            predicted_today_price = generate_future_predictions(model, scaler, last_100_days, 1)[0]
            return predicted_today_price * multiplying_factors[stock_name]
        else:
            # Show today's actual price (market closed)
            return historical_data.iloc[-1]['Close']

    else:
        # Predict for future dates
        num_days = (selected_date - today).days
        historical_data = historical_data.tail(100)
        last_100_days = historical_data['Close'].values.reshape(-1, 1)
        future_predictions = generate_future_predictions(model, scaler, last_100_days, num_days)
        return future_predictions[-1] * multiplying_factors[stock_name]

# Display Prediction
prediction = get_stock_prediction(stock_name, selected_date)

if prediction is not None:
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-value">${prediction:.2f}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-text">Predicted Price for {stock_name} on {selected_date}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)