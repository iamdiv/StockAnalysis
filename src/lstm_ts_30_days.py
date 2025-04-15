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

def calculate_rmse(model, scaler,X_test_tensor,y_test_tensor):
    # Evaluate on test data
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        predicted_test = model(X_test_tensor)

    # Convert tensors to numpy
    predicted_test_np = predicted_test.numpy()
    y_test_np = y_test_tensor.numpy()

    # Create dummy array to match scaler's expected shape
    dummy_pred = np.zeros((predicted_test_np.shape[0], 3))
    dummy_real = np.zeros((y_test_np.shape[0], 3))

    # Only fill the 'Close' price (assuming it's the first feature)
    dummy_pred[:, 0] = predicted_test_np[:, 0]  # first day of forecast for each sequence
    dummy_real[:, 0] = y_test_np[:, 0]

    # Inverse transform using the scaler
    predicted_prices = scaler.inverse_transform(dummy_pred)[:, 0]
    actual_prices = scaler.inverse_transform(dummy_real)[:, 0]

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_np.flatten(), predicted_test_np.flatten()))
    return rmse

class StockPricePredictor(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size = 100, num_layers=2, forcasted_horizon=30):
        super(StockPricePredictor, self).__init__()

        #LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        #Dropout layer
        self.dropout = nn.Dropout(0.2)

        #fully Connected Layer
        self.fc = nn.Linear(hidden_layer_size, forcasted_horizon)

    def forward(self, x):
        out,(hn,cn) = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:,-1,:])
        return out

#create a time step data structure 
def create_dataset(data, time_step=60, forecast_horizon=30):
    X,y = [],[]
    for i in range(time_step, len(data)-forecast_horizon+1):
       
        X.append(data[i-time_step:i,:])
        y.append(data[i:i+forecast_horizon,0])

    return np.array(X), np.array(y)

def prepare_data(ticker):
    #plot historical data.
    start_date = datetime(2001, 1, 1) 
    end_date = datetime.today().date()
    historical_data = si.get_historical_data(start_date=start_date, end_date=end_date, ticker=ticker)
    data = historical_data[[f'Close_{ticker}']]

    #calculate moving avg 
    data['EWM50'] = data[f'Close_{ticker}'].ewm(span=50, adjust=False).mean()
    data['EWM20'] = data[f'Close_{ticker}'].ewm(span=20, adjust=False).mean()

    # Drop NaN values (caused by the moving averages)
    data = data.dropna()

    #Normalize the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    #Preparing data for training (using 60 days for predicting the nest day's price)
    X,y = create_dataset(scaled_data, time_step=60)

    #Reshape the data for LSTM [Samples, time steps , features]
    X = X.reshape(X.shape[0], X.shape[1],X.shape[2])

    #split the data into the training an test sets(80% Train and 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

    #convert to pytorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor, scaler

def train_model(ticker):

    #load data
    X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor, scaler = prepare_data(ticker)
    #initialize the model, Define Loss and optimiser 
    model = StockPricePredictor(input_size=3, hidden_layer_size=100, num_layers=2,forcasted_horizon=30)

    #Loss Funtion (Mean Squared Error)
    criterion = nn.MSELoss()

    #Optimiser(Adam)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    #set the number of epochs
    epochs = 150
    
    for epoch in range(epochs):
        #forward pass
        outputs = model(X_train_tensor)

        #calculate loss
        loss = criterion(outputs, y_train_tensor)

        #apply backpropoagation and optimization
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (epoch + 1)%5 == 0: #print Every 2 epochs 
            print(f"Epochs[{epoch+1}/{epochs}], Loss:{loss.item():.4f}")

    rmse_score = calculate_rmse(model, scaler,X_test_tensor,y_test_tensor)

    print(f"\nTest RMSE: {rmse_score:.4f}")

    #Save Model 
    #Save the trained model
    torch.save(model.state_dict(), f'models/{ticker}_model.pth')

    # Save the scaler
    joblib.dump(scaler, f'models/{ticker}_scaler.pkl')

    print("Model and Scaler saved successfully.")
