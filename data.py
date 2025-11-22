import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

# can change later
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
num_assets = len(tickers)
# need to decide start year
raw = yf.download(
    tickers, 
    start="2020-01-01", 
    auto_adjust=True
)
data = raw["Close"]

print(data.head())


# daily returns, drop first row b/c DNE
returns = raw.pct_change().dropna()

print(returns.head())

# get market caps
market_caps = []
for t in tickers:
    # get ticker's metadata
    info = yf.Ticker(t).info
    market_caps.append(info["marketCap"])
market_caps = np.array(market_caps) # make it a numpy array
# normalize so that market caps sum to 1 (aka portfolio weights)
cap_weights = market_caps / market_caps.sum()
# get cap-weighted portfolio returns
cap_portfolio_returns = returns.dot(cap_weights)


