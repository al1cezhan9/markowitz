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




