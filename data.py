import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

# List of tickers from many sectors
# These will be used to download price data and run portfolio analysis
tickers = [
    "WMT", "XOM", "MSFT", "F", "GE", "COP", "C", "AIG", 
    "IBM", "GOOGL", "AAPL", "HD", "VZ", "MCK", 
    "CAH", "MO", "BAC", "JPM", "KR", "VLO", "COR", "PFE", 
    "NEE", "DUK", "LIN", "PLD"
]

# Download historical daily prices using yfinance
# auto_adjust=True adjusts for splits and dividends for clean returns
raw = yf.download(tickers, start="2005-01-01", auto_adjust=True, progress=False)
# We only need the closing prices
data = raw["Close"]

# Check percentage of missing values per ticker
missing_pct = data.isnull().sum() / len(data)*100
for ticker in data.columns:
    pct = missing_pct[ticker]
    if pct > 0:
        print(f"{ticker}: {pct:.1f}% missing")

# Forward fill and backward fill to handle missing prices
# (Usually needed if a stock IPO's later or has sparse data)
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')

# Update tickers list after cleaning
tickers = data.columns.tolist()
num_assets = len(tickers)

# Compute **daily** returns and drop NaNs (first row becomes NaN)
returns = data.pct_change().dropna()
# Convert daily returns to annualized expected return
mu = returns.mean() * 252
# Compute annualized covariance matrix of returns
cov = returns.cov() * 252

print("Expected returns (annualized):")
print(mu.sort_values(ascending=False).head(5))
print(f"\nRange: [{mu.min():.1%}, {mu.max():.1%}]\n")

# Force covariance matrix symmetry (minor numerical fix)
cov_matrix = cov.values.copy()
cov_matrix = (cov_matrix + cov_matrix.T) / 2

# Compute eigenvalues to inspect positive-semidefinite nature of covariance
# PS: Some solvers fail if matrix isn't PSD
eigenvalues = np.linalg.eigvalsh(cov_matrix)
print(f"Covariance eigenvalues: [{eigenvalues.min():.2e}, {eigenvalues.max():.2e}]")

# Function to compute minimum volatility portfolio achieving a target return
def min_vol_for_target_return(target_ret):
    # Optimization variable: portfolio weights
    w = cp.Variable(num_assets)

    # Objective: minimize portfolio variance
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))

    # Constraints:
    # 1) fully invested (weights sum to 1)
    # 2) long-only (no shorting)
    # 3) portfolio achieves specific return
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w @ mu.values == target_ret
    ]

    prob = cp.Problem(objective, constraints)

    try:
        # OSQP is fast for quadratic problems
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return w.value
        else:
            return None
    except Exception as e:
        return None

# Build the efficient frontier by sweeping target returns
ret_min = mu.min()
ret_max = mu.max()
target_returns = np.linspace(ret_min, ret_max, 40)
frontier_vol = []
frontier_ret = []

for i, tr in enumerate(target_returns):
    w_opt = min_vol_for_target_return(tr)
    # Compute portfolio volatility from optimized weights
    vol = np.sqrt(w_opt.T @ cov_matrix @ w_opt)
    frontier_vol.append(vol)
    frontier_ret.append(tr)

print(f"{len(frontier_vol)}/{len(target_returns)} frontier points computed\n")

# Monte‑Carlo simulation of random portfolios for comparison
num_portfolios = 5000
random_returns = []
random_vols = []

for _ in range(num_portfolios):
    # Generate random weights and normalize to sum to 1
    w = np.random.rand(num_assets)
    w = w / w.sum()

    # Compute return and volatility for the random portfolio
    r = float(w @ mu.values)
    v = float(np.sqrt(w.T @ cov_matrix @ w))

    random_returns.append(r)
    random_vols.append(v)

# Equal‑weight portfolio (EW)
equal_weights = np.ones(num_assets) / num_assets
ew_portfolio_returns = returns.dot(equal_weights)

# Market‑cap weighted portfolio (CAP)
# yfinance 'info' returns metadata including market cap
market_caps = []
for t in tickers:
    try:
        info = yf.Ticker(t).info
        mc = info.get("marketCap", None)
        market_caps.append(mc if mc else 0)
    except:
        market_caps.append(0)

market_caps = np.array(market_caps, dtype=float)
# Normalize market caps into portfolio weights
cap_weights = market_caps / market_caps.sum()
# Compute CAP portfolio daily returns
cap_portfolio_returns = returns.dot(cap_weights)

# Minimum‑variance portfolio without return constraint (MVO)
w = cp.Variable(num_assets)
objective = cp.Minimize(cp.quad_form(w, cov_matrix))
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.OSQP, verbose=False)

mvo_weights = w.value
mvo_portfolio_returns = returns.dot(mvo_weights)

# Utility functions for performance metrics

def annualized_return(ret):
    # Convert cumulative daily returns to annualized rate
    return (1 + ret).prod() ** (252 / len(ret)) - 1

def annualized_volatility(ret):
    return ret.std() * np.sqrt(252)

def sharpe_ratio(ret, rf_rate=0.037):
    # Convert annual risk‑free rate to daily
    rf_daily = rf_rate / 252
    excess = ret - rf_daily
    return np.sqrt(252) * (excess.mean() / excess.std())

def max_drawdown(ret):
    cum_returns = (1 + ret).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

# Build summary performance table for CAP / MVO / EW
results = pd.DataFrame({
    "CAP": [
        annualized_return(cap_portfolio_returns),
        annualized_volatility(cap_portfolio_returns),
        sharpe_ratio(cap_portfolio_returns),
        max_drawdown(cap_portfolio_returns)
    ],
    "MVO": [
        annualized_return(mvo_portfolio_returns),
        annualized_volatility(mvo_portfolio_returns),
        sharpe_ratio(mvo_portfolio_returns),
        max_drawdown(mvo_portfolio_returns)
    ],
    "EW": [
        annualized_return(ew_portfolio_returns),
        annualized_volatility(ew_portfolio_returns),
        sharpe_ratio(ew_portfolio_returns),
        max_drawdown(ew_portfolio_returns)
    ]
}, index=["Annual Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown"])

print(results.to_string())

# Plot efficient frontier + CAP + MVO + EW
plt.plot(frontier_vol, frontier_ret, 'b-', label="Efficient Frontier", linewidth=2, color="grey")
plt.scatter(annualized_volatility(cap_portfolio_returns), 
          annualized_return(cap_portfolio_returns), 
          c="red", s=150, marker='o', label="CAP", zorder=5, edgecolors='black')
plt.scatter(annualized_volatility(mvo_portfolio_returns), 
          annualized_return(mvo_portfolio_returns), 
          c="blue", s=150, marker='s', label="MVO", zorder=5, edgecolors='black')
plt.scatter(annualized_volatility(ew_portfolio_returns), 
          annualized_return(ew_portfolio_returns), 
          c="green", s=150, marker='^', label="EW", zorder=5, edgecolors='black')
plt.xlabel("Volatility (Annual Std. Dev)", fontsize=11)
plt.ylabel("Expected Return (Annual)", fontsize=11)
plt.title("Efficient Frontier", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.savefig("efffront.png")
plt.show()

# Compute cumulative returns for CAP / MVO / EW
cum_cap = (1 + cap_portfolio_returns).cumprod()
cum_mvo = (1 + mvo_portfolio_returns).cumprod()
cum_ew = (1 + ew_portfolio_returns).cumprod()

plt.plot(cum_cap.index, cum_cap.values, label="CAP", linewidth=2, color="red")
plt.plot(cum_mvo.index, cum_mvo.values, label="MVO", linewidth=2, color="blue")
plt.plot(cum_ew.index, cum_ew.values, label="EW", linewidth=2, color="green")
plt.title("Cumulative Returns", fontsize=12, fontweight='bold')
plt.xlabel("Date", fontsize=11)
plt.ylabel("Growth of $1", fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig("cumret.png")
plt.show()

# Sort and print holdings of MVO portfolio
mvo_holdings = pd.Series(mvo_weights, index=tickers).sort_values(ascending=False)
print("MVO Portfolio Holdings")
print(mvo_holdings.to_string() + "\n")

# Sort and print holdings of CAP portfolio
cap_holdings = pd.Series(cap_weights, index=tickers).sort_values(ascending=False)
print("CAP Portfolio Holdings")
print(cap_holdings.to_string() + "\n")

# Compute rolling one‑year (252‑day) volatility for all portfolios
window = 252
rolling_vol_cap = cap_portfolio_returns.rolling(window).std() * np.sqrt(window)
rolling_vol_mvo = mvo_portfolio_returns.rolling(window).std() * np.sqrt(window)
rolling_vol_ew = ew_portfolio_returns.rolling(window).std() * np.sqrt(window)

plt.plot(rolling_vol_cap.index, rolling_vol_cap.values, label="CAP", linewidth=2, color="red")
plt.plot(rolling_vol_mvo.index, rolling_vol_mvo.values, label="MVO", linewidth=2, color="blue")
plt.plot(rolling_vol_ew.index, rolling_vol_ew.values, label="EW", linewidth=2, color="green")
plt.title("Rolling 1-Year Volatility", fontsize=12, fontweight='bold')
plt.ylabel("Annualized Volatility", fontsize=11)
plt.xlabel("Date", fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig("rollvol.png")
plt.show()



