import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime,timedelta
from src import stock_info as si
import joblib
from sklearn.metrics import mean_squared_error
import os
from src.lstm_ts_30_days import StockPricePredictor

def load_data(ticker,days):
    #plot historical data.
    end_date = datetime.today()
    start_date = end_date - timedelta(days=150) 

    data = si.get_historical_data(start_date=start_date, end_date=end_date, ticker=ticker)
    data = data[[f'Close_{ticker}']]

    #calculate exponential moving avg 
    data['EWM50'] = data[f'Close_{ticker}'].ewm(span=50, adjust=False).mean()
    data['EWM20'] = data[f'Close_{ticker}'].ewm(span=20, adjust=False).mean()

    df_sorted = data.sort_index(ascending=False)
    df_sorted = df_sorted.head(days)

    return df_sorted

def predict_next_30_days(model, scaler, data, device='cpu'):
    """
    Predict the next 30 days of stock prices using the trained model.

    Parameters:
        model: Trained PyTorch model
        scaler: Scaler used for normalization (MinMaxScaler)
        data: DataFrame with the last known input data (at least 60 rows, with 3 features)
        device: 'cpu' or 'cuda'

    Returns:
        np.array: Array of predicted prices (shape: 30,)
    """

    model.eval()  # set model to eval mode

    # Get the last 60 rows
    last_60_days = data[-60:].values

    # Scale it using the same scaler used for training
    last_60_scaled = scaler.transform(last_60_days)

    # Reshape to (1, 60, 3) for LSTM
    input_seq = torch.tensor(last_60_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_seq).cpu().numpy()

    # The model outputs 30 normalized prices — inverse transform needed
    # Create dummy array to match scaler's expected shape
    dummy = np.zeros((30, last_60_days.shape[1]))
    dummy[:, 0] = prediction[0]  # only setting the Close price (index 0)

    # Inverse transform
    predicted_prices = scaler.inverse_transform(dummy)[:, 0]  # Get only the predicted 'Close' prices

    return predicted_prices

def predictor(classifier_str,ticker):
    if classifier_str == "monthlyforcast":
        # Instantiate and load the model
        model = StockPricePredictor(input_size=3, hidden_layer_size=100, num_layers=2,forcasted_horizon=30)
        model.load_state_dict(torch.load(f'models/{ticker}_model.pth'))
        # Load the scaler
        scaler = joblib.load(f'models/{ticker}_scaler.pkl')
        data = load_data(ticker=ticker, days=60)
        predicted_prices = predict_next_30_days(model, scaler, data, device='cpu')

    return predicted_prices, data

