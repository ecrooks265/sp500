# Load necessary libraries
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime, timedelta

# Function to fetch today's data with technical indicators
def fetch_today_data(ticker):
    # Extend the start date to ensure enough data points
    start_date = datetime.now() - timedelta(days=365)
    
    # Fetch historical data for the ticker
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
    
    if data.empty:
        print(f"No data found for {ticker}")
        return None
    
    data['Ticker'] = ticker
    data['Date'] = data.index  # Add Date column
    data = data.rename(columns={'Adj Close': 'Adj_Close'})
    
    # Only calculate indicators if there are enough data points
    if len(data) < 200:
        print(f"Not enough data points for {ticker}")
        return data

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

    return data

# Example usage for predicting tomorrow's performance
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "ADBE", "NFLX", "CSCO",
    "INTC", "ORCL", "IBM", "CRM", "PYPL",
    "QCOM", "TXN", "AVGO", "AMD", "INTU",
    "SHOP", "SNAP", "SQ", "UBER"]

all_data = pd.DataFrame()

for ticker in tickers:
    ticker_data = fetch_today_data(ticker)
    if ticker_data is not None:
        all_data = pd.concat([all_data, ticker_data])

# Extract today's metrics for prediction (Example: using last row of each ticker)
today_metrics = all_data.groupby('Ticker').last().reset_index()

# Select relevant columns for prediction
features = ['SMA_50', 'SMA_200', 'EMA_50', 'EMA_200', 'RSI', 'Stochastic_Oscillator',
            'MACD', 'MACD_Signal', 'CCI', 'Bollinger_High', 'Bollinger_Low', 'ATR',
            'Volume_SMA_20', 'Volatility', 'Month', 'Day_of_Week']
today_metrics = today_metrics[features]

# Save to CSV
all_data.to_csv("sp500_test_data.csv", index=False)
