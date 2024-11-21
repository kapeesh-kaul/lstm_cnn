import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker, start, end):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    data = data[['Close']].reset_index()
    return data

def preprocess_data(data):
    """Normalize data using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data['Close'] = scaler.fit_transform(data[['Close']])
    return data, scaler

def create_sequences(data, seq_length):
    """Generate sequences for LSTM input."""
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)
