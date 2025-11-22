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
    start="2000-01-01", 
    auto_adjust=True
)
data = raw["Close"]

# print(data.head())

# daily returns, drop first row b/c DNE
returns = data.pct_change().dropna()

# print(returns.head())

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


# this is the markowitz mean variance optimizing part
cov = returns.cov().values # cov matrix of returns
w = cp.Variable(num_assets) # optimization variables (one per asset)
objective = cp.Minimize(cp.quad_form(w, cov)) # minimize portfolio variance aka w^T Σ w
constraints = [
    cp.sum(w) == 1,
    w >= 0
]
problem = cp.Problem(objective, constraints) # optimize!
problem.solve() # LOL if only i could call this fxn on my life

# extract optimal weights 
mvo_weights = w.value
# daily returns of optimized portfolio
mvo_portfolio_returns = returns.dot(mvo_weights)

