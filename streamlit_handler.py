# import module
import streamlit as st
from src import stock_info as si
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.express as px
import mplcursors

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

