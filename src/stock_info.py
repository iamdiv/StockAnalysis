import requests
from bs4 import BeautifulSoup
import yfinance as yf
from src import constants
import pandas as pd

def find_real_time_stocks(ticker, exchange):
    nse_url = f"https://www.google.com/finance/quote/{ticker}:{exchange}"
    response = requests.get(nse_url)
    soup = BeautifulSoup(response.text,'html.parser')
    stock_price_class = "YMlKec fxKbKc"
    real_time_price = soup.find(class_ = stock_price_class).text.strip()[1:].replace(",","")
    return real_time_price
    
def get_historical_data(start_date, end_date, ticker):
    
    data = yf.download(ticker, start = start_date, 
                   end = end_date) 
    print(f"data shape check:{data.shape}")
    # Flatten the MultiIndex columns
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

    return data

def extract_stock_info(ticker):
    stock_info = yf.Ticker(ticker).info
    stock_info_dict = {key: stock_info[key] for key in constants.stock_info_list}
    stock_info_df = pd.DataFrame([stock_info_dict])
    return stock_info_df
