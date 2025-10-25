# app_lstm_web.py

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

st.set_page_config(page_title="LSTM Stock Price Prediction", page_icon="📈")

st.title("📈 LSTM Stock Price Prediction Web App")
st.write("Predict stock prices using LSTM (Deep Learning)")

# Add banner image
st.image(
    "https://images.unsplash.com/photo-1603791440384-56cd371ee9a7?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxNjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-4.0.3&q=80&w=1080",
    use_container_width=True
)

# Input stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")

if ticker:
    st.write(f"Fetching data for: {ticker}")
    
    # Download historical stock data
    data = yf.download(ticker, start="2015-01-01", end="2025-12-31")
    st.subheader("Data preview:")
    st.dataframe(data.head())

    st.write("Training LSTM model... This may take a few minutes.")
    
    # Prepare data
    data_close = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data_close)

    seq_length = 60
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,  # Reduced epochs for demo speed
        batch_size=32,
        verbose=1
    )

    # Predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

    # Plot results
    st.subheader("Stock Price Prediction vs Actual Price")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_test_actual, label="Actual Price")
    ax.plot(predictions, label="Predicted Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.success("Prediction completed!")
