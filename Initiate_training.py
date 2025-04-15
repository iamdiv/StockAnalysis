from src.lstm_ts_30_days import train_model 

if __name__ == "__main__":
    ticker_ls = ["^NSEI","LEMONTREE.NS","SBIN.NS"]
    for ticker in ticker_ls:
        print(f"Model Training Started for : {ticker}")
        train_model(ticker)
