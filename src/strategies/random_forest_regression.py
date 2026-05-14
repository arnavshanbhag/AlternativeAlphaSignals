import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]

data = yf.download(tickers, start="2025-01-01")["Close"]

returns = data.pct_change()
momentum_5 = data.pct_change(5)
momentum_20 = data.pct_change(20)
vol_20 = returns.rolling(20).std()
ma_20 = data.rolling(20).mean()
ma_gap = data / ma_20 - 1
target = returns.shift(-1)

df = []

for ticker in tickers:
    tmp = pd.DataFrame({
        "return": returns[ticker],
        "mom_5": momentum_5[ticker],
        "mom_20": momentum_20[ticker],
        "vol_20": vol_20[ticker],
        "ma_gap": ma_gap[ticker],
        "target": target[ticker]
    })

    tmp["ticker"] = ticker
    df.append(tmp)

df = pd.concat(df)
df = df.dropna()

df[df["ticker"] == "MSFT"].head()

features = ["return", "mom_5", "mom_20", "vol_20", "ma_gap"]

X = df[features]
y = df["target"]

# ----------------------------
# 1. TIME SPLIT
# ----------------------------

split = int(0.8 * len(df))

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ----------------------------
# 2. MODEL
# ----------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# 3. PREDICTIONS
# ----------------------------

y_pred = model.predict(X_test)

# ----------------------------
# 4. EVALUATION (SIGNAL QUALITY)
# ----------------------------

corr = np.corrcoef(y_pred, y_test)[0, 1]
print("Prediction correlation:", corr)

mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# ----------------------------
# 5. SIMPLE TRADING STRATEGY
# ----------------------------

signal = np.where(y_pred > 0, 1, -1)
strategy_returns = signal * y_test.values

cumulative = np.cumprod(1 + strategy_returns)

plt.plot(cumulative)
plt.title("Simple Long/Short Strategy Performance")
plt.show()