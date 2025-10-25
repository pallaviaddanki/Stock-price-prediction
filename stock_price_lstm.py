# stock_price_simple.py
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Download stock data
ticker = "AAPL"
start_date = "2015-01-01"
end_date = "2025-12-31"

data = yf.download(ticker, start=start_date, end=end_date)
data = data[['Close']]
print(data.head())

# 2. Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
SEQ_LEN = 60
X, y = [], []
for i in range(SEQ_LEN, len(scaled_data)):
    X.append(scaled_data[i-SEQ_LEN:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), y_train)

# 5. Make predictions
predictions = model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1]))
predictions = scaler.inverse_transform(predictions.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# 6. Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, color='blue', label='Actual Price')
plt.plot(predictions, color='red', label='Predicted Price')
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
