# import module
import streamlit as st
from src import stock_info as si
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.express as px
import mplcursors
import matplotlib.pyplot as plt
from src.future_prediction import predictor

def plot_forecast(classifier_str,ticker):
    """
    Plots the last 60 days of actual prices and the next 30 days of predicted prices.

    Parameters:
        data: Original DataFrame with the 'Close' column
        predicted_prices: Array of predicted closing prices (length = 30)
    """
    predicted_prices,last_60_actual = predictor(classifier_str,ticker)
    last_60_actual = last_60_actual[f'Close_{ticker}'].values
    total_days = np.arange(1, 91)
    full_series = np.concatenate([last_60_actual, predicted_prices])

    fig = plt.figure(figsize=(12, 6))
    plt.plot(total_days, full_series, label='Price (Actual + Predicted)', color='darkorange')

    # Optional: Add shaded background to distinguish actual vs predicted
    plt.axvspan(1, 60, color='lightblue', alpha=0.3, label='Actual')
    plt.axvspan(60, 90, color='lightgreen', alpha=0.2, label='Predicted')

    plt.title(f"Stock Price Forecast for Close_{ticker}")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    

    return fig

# Title
st.title("Stocks Monitoring Dashboards")

ticker = "SBIN.NS"
df = si.extract_stock_info(ticker)

# Header
st.header(f"Information for {ticker} Ticker in INR") 

st.table(df) 

#plot historical data.
start_date = datetime(2025, 1, 1) 
end_date = datetime.today().date()

historical_data = si.get_historical_data(start_date=start_date, end_date=end_date, ticker=ticker)
print(historical_data.head())
# Create a Plotly scatter plot
fig = px.line(historical_data, x=historical_data.index, y=[f'Close_{ticker}', f'High_{ticker}', f'Low_{ticker}'], title=f'{ticker} Stock Prices')

fig.update_traces(hovertemplate='Date: %{x}<br>Price: %{y}<extra></extra>')

# Display the plot in Streamlit
st.plotly_chart(fig)

# Streamlit UI elements
classifier_str = "monthlyforcast"
fig = plot_forecast(classifier_str,ticker)
st.title(f"Predicted Trend for {ticker}")
st.pyplot(fig)

#Plot Trend for NIFTY 50
fig = plot_forecast(classifier_str,"^NSEI")
st.title(f"Predicted Trend for NIFTY50 for next 30 days")
st.pyplot(fig)