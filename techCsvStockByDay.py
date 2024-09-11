import yfinance as yf
import pandas as pd
import ta
from datetime import datetime

# List of S&P 500 companies (tickers)
sp500_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "ADBE", "NFLX", "CSCO",
    "INTC", "ORCL", "IBM", "CRM", "PYPL",
    "QCOM", "TXN", "AVGO", "AMD", "INTU",
    "SHOP", "SNAP", "SQ", "UBER"
]

# Function to fetch data for a list of tickers
def fetch_data(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        data = yf.download(ticker, start=start_date, end=end_date)
        data['Ticker'] = ticker
        data['Date'] = data.index  # Add Date column
        data = data.rename(columns={'Adj Close': 'Adj_Close'})

        # Calculate technical indicators using ta library
        # Moving Averages
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
        data['EMA_50'] = ta.trend.ema_indicator(data['Close'], window=50)
        data['EMA_200'] = ta.trend.ema_indicator(data['Close'], window=200)

        # Momentum Indicators
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        data['Stochastic_Oscillator'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
        data['MACD'] = ta.trend.macd(data['Close'])
        data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])
        data['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'])

        # Volatility Indicators
        data['Bollinger_High'] = ta.volatility.bollinger_hband(data['Close'])
        data['Bollinger_Low'] = ta.volatility.bollinger_lband(data['Close'])
        data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])

        # Volume Indicators
        data['Volume_SMA_20'] = ta.trend.sma_indicator(data['Volume'], window=20)

        # Additional financial metrics
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=21).std()  # 21-day rolling window for monthly volatility
        
        # Add date-related features for seasonality
        data['Month'] = data.index.month
        data['Day_of_Week'] = data.index.dayofweek

        # Shift the target variable (next day's return) to create training label
        data['Next_Day_Return'] = data['Daily_Return'].shift(-1)

        all_data.append(data)
    return pd.concat(all_data)

# Fetch data from 1999 to the current date
start_date = "1999-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')
data = fetch_data(sp500_tickers, start_date, end_date)

# Save to CSV
data.to_csv("sp500_data_with_indicators.csv", index=False)
