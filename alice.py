import numpy as np
import pandas as pd
import yfinance as yf
import cvxpy as cp
import matplotlib.pyplot as plt

tickers = [
    "WMT", "XOM", "MSFT", "F", "GE", "COP", "C", "AIG", 
    "IBM", "GOOGL", "AAPL", "HD", "VZ", "MCK", 
    "CAH", "MO", "BAC", "JPM", "KR", "VLO", "COR", "PFE"
]

raw = yf.download(tickers, start="2005-01-01", auto_adjust=True, progress=False)
data = raw["Close"]

# missing_pct = data.isnull().sum() / len(data)*100
# for ticker in data.columns:
#     pct = missing_pct[ticker]
#     if pct > 0:
#         print(f"  {ticker}: {pct:.1f}% missing")

# forward/backwards fill? no longer necessary
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')

tickers = data.columns.tolist()
num_assets = len(tickers)

# daily returns
returns = data.pct_change().dropna()
# annualize
mu = returns.mean() *252
cov = returns.cov() *252

print("Expected returns (annualized):")
print(mu.sort_values(ascending=False).head(5))
print(f"\nRange: [{mu.min():.1%}, {mu.max():.1%}]\n")

# force sym
cov_matrix = cov.values.copy()
cov_matrix = (cov_matrix + cov_matrix.T) / 2

# # positive semi-definite
eigenvalues = np.linalg.eigvalsh(cov_matrix)
# min_eig = eigenvalues.min()
# # if min_eig < 1e-6:
# #     print(f"Regularizing covariance matrix (min eigenvalue: {min_eig:.2e})")
# #     cov_matrix += np.eye(num_assets) * max(abs(min_eig) * 1.1, 1e-6)
# #     eigenvalues = np.linalg.eigvalsh(cov_matrix)

print(f"Covariance eigenvalues: [{eigenvalues.min():.2e}, {eigenvalues.max():.2e}]")

def min_vol_for_target_return(target_ret):
    w = cp.Variable(num_assets)
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w @ mu.values == target_ret
    ]
    prob = cp.Problem(objective, constraints)
    
    try: # shouldn't be necessary i think
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return w.value
        else:
            return None
    except Exception as e:
        return None

# gridsearch
ret_min = mu.min()
ret_max = mu.max()
target_returns = np.linspace(ret_min, ret_max, 40)
frontier_vol = []
frontier_ret = []

for i, tr in enumerate(target_returns):
    w_opt = min_vol_for_target_return(tr)
    vol = np.sqrt(w_opt.T @ cov_matrix @ w_opt)
    frontier_vol.append(vol)
    frontier_ret.append(tr) 

print(f"{len(frontier_vol)}/{len(target_returns)} frontier points computed\n")

equal_weights = np.ones(num_assets) / num_assets
ew_portfolio_returns = returns.dot(equal_weights)

market_caps = []
for t in tickers:
    try:
        info = yf.Ticker(t).info
        mc = info.get("marketCap", None)
        market_caps.append(mc if mc else 0)
    except:
        market_caps.append(0)

market_caps = np.array(market_caps, dtype=float)
cap_weights = market_caps / market_caps.sum()
cap_portfolio_returns = returns.dot(cap_weights)

w = cp.Variable(num_assets)
objective = cp.Minimize(cp.quad_form(w, cov_matrix))
constraints = [cp.sum(w) == 1, w >= 0]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.OSQP, verbose=False)

mvo_weights = w.value
mvo_portfolio_returns = returns.dot(mvo_weights)

def annualized_return(ret):
    return (1 + ret).prod() ** (252 / len(ret)) - 1

def annualized_volatility(ret):
    return ret.std() * np.sqrt(252)

def sharpe_ratio(ret, rf_rate=0.037):
    rf_daily = rf_rate / 252
    excess = ret - rf_daily
    return np.sqrt(252) * (excess.mean() / excess.std())

def max_drawdown(ret):
    cum_returns = (1 + ret).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

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

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
ax = axes[0, 0]
ax.plot(frontier_vol, frontier_ret, 'b-', label="Efficient Frontier", linewidth=2)
ax.scatter(annualized_volatility(cap_portfolio_returns), 
          annualized_return(cap_portfolio_returns), 
          c="red", s=150, marker='o', label="CAP", zorder=5, edgecolors='black')
ax.scatter(annualized_volatility(mvo_portfolio_returns), 
          annualized_return(mvo_portfolio_returns), 
          c="blue", s=150, marker='s', label="MVO", zorder=5, edgecolors='black')
ax.scatter(annualized_volatility(ew_portfolio_returns), 
          annualized_return(ew_portfolio_returns), 
          c="purple", s=150, marker='^', label="EW", zorder=5, edgecolors='black')
ax.set_xlabel("Volatility (Annual Std. Dev)", fontsize=11)
ax.set_ylabel("Expected Return (Annual)", fontsize=11)
ax.set_title("Efficient Frontier", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

ax = axes[0, 1]
cum_cap = (1 + cap_portfolio_returns).cumprod()
cum_mvo = (1 + mvo_portfolio_returns).cumprod()
cum_ew = (1 + ew_portfolio_returns).cumprod()
ax.plot(cum_cap.index, cum_cap.values, label="CAP", linewidth=2)
ax.plot(cum_mvo.index, cum_mvo.values, label="MVO", linewidth=2)
ax.plot(cum_ew.index, cum_ew.values, label="EW", linewidth=2)
ax.set_title("Cumulative Returns", fontsize=12, fontweight='bold')
ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Growth of $1", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# ax = axes[1, 0]
# mvo_holdings = pd.Series(mvo_weights, index=tickers).sort_values(ascending=True)
# top_holdings = mvo_holdings.tail(10)
# colors = ['#2ecc71' if x > 0.05 else '#3498db' for x in top_holdings]
# top_holdings.plot(kind='barh', ax=ax, color=colors)
# ax.set_title("MVO Portfolio - Top 10 Holdings", fontsize=12, fontweight='bold')
# ax.set_xlabel("Portfolio Weight", fontsize=11)
# ax.grid(True, alpha=0.3, axis='x')

# ax = axes[1, 1]
# window = 252
# rolling_vol_cap = cap_portfolio_returns.rolling(window).std() * np.sqrt(252)
# rolling_vol_mvo = mvo_portfolio_returns.rolling(window).std() * np.sqrt(252)
# rolling_vol_ew = ew_portfolio_returns.rolling(window).std() * np.sqrt(252)

# ax.plot(rolling_vol_cap.index, rolling_vol_cap.values, label="CAP", linewidth=2)
# ax.plot(rolling_vol_mvo.index, rolling_vol_mvo.values, label="MVO", linewidth=2)
# ax.plot(rolling_vol_ew.index, rolling_vol_ew.values, label="EW", linewidth=2)
# ax.set_title("Rolling 1-Year Volatility", fontsize=12, fontweight='bold')
# ax.set_ylabel("Annualized Volatility", fontsize=11)
# ax.set_xlabel("Date", fontsize=11)
# ax.legend(fontsize=10)
# ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
