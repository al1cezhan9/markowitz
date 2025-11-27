import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

# (can change later), UPDATE: added more tickers to diversify better
# is this too few? too many? are they representative enough?
tickers = ["XOM", "MSFT", "C", "GE", "WMT", "BAC", "JNJ", "PFE", 
           "INTC", "AIG", "IBM", "PG", "BRK-B", "KO", "MRK", 
           "DIS", "MO", "CVX", "CSCO", "T", "AMZN", "GOOGL", "NVDA"
        ]




num_assets = len(tickers)
# need to decide start year
raw = yf.download(
    tickers, 
     #change to earlier to avoid survivorship bias and to include 2008 crisis data
     #BUT this means that some companies like META can't be included since they IPOed later
     #look at what other papers tend to use to decide how many years back to go
    start="2005-01-01", 
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

# analyze performance

# how much the portfolio grew per year on average
def annualized_return(ret):
    return (1 + ret).prod() ** (252 / len(ret)) - 1

# how much the portfolio fluctuated day-to-day
def annualized_volatility(ret):
    return ret.std() * np.sqrt(252)

# table for comparison
results = pd.DataFrame({
    "CAP": [
        annualized_return(cap_portfolio_returns),
        annualized_volatility(cap_portfolio_returns)
    ],
    "MVO": [
        annualized_return(mvo_portfolio_returns),
        annualized_volatility(mvo_portfolio_returns)
    ]
},
index=["Annual Return", "Annual Volatility"])

print("\nPortfolio Comparison:")
print(results)

# cum growth of $1 invested
cum_cap = (1 + cap_portfolio_returns).cumprod()
cum_mvo = (1 + mvo_portfolio_returns).cumprod()

# now plot
plt.plot(cum_cap, label="CAP")
plt.plot(cum_mvo, label="MVO")
plt.title("Cumulative Return Comparison")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True)
plt.show()

# i think this has to be adjusted based on real historical numbers??? argh 
risk_free_rate = 0.037  # nominal should be used, .037 bc avg nom from 2005 to now
# if switch to 2015 start use .025
rf_daily = risk_free_rate / 252
excess_daily_return_cap = cap_portfolio_returns - rf_daily
excess_daily_return_mvo = mvo_portfolio_returns - rf_daily


# sharpe is basically risk-adjusted return of an investment
# = excess return over risk-free rate, divided by stdev
sharpe_ratio_cap = np.sqrt(252) * (excess_daily_return_cap.mean() / excess_daily_return_cap.std())
sharpe_ratio_mvo = np.sqrt(252) * (excess_daily_return_mvo.mean() / excess_daily_return_mvo.std())
print("\nSharpe Ratio (CAP):", sharpe_ratio_cap)
print("Sharpe Ratio (MVO):", sharpe_ratio_mvo)
