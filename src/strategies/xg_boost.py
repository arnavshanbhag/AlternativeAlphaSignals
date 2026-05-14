import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]

data = yf.download(tickers, start="2025-01-01")["Close"]

returns = data.pct_change()
momentum_5 = data.pct_change(5)
momentum_20 = data.pct_change(20)
vol_20 = returns.rolling(20).std()
ma_20 = data.rolling(20).mean()
ma_gap = data / ma_20 - 1

# predict next-day return
target = returns.shift(-1)

df_list = []

for t in tickers:
    tmp = pd.DataFrame({
        "return": returns[t],
        "mom_5": momentum_5[t],
        "mom_20": momentum_20[t],
        "vol_20": vol_20[t],
        "ma_gap": ma_gap[t],
        "target": target[t]
    })
    
    tmp["ticker"] = t
    df_list.append(tmp)

df = pd.concat(df_list).dropna()
df = df.sort_index()

features = ["return", "mom_5", "mom_20", "vol_20", "ma_gap"]

X = df[features]
y = df["target"]

split = int(0.8 * len(df))

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

corr = np.corrcoef(y_pred, y_test)[0, 1]
print("Prediction correlation:", corr)

signal = np.where(y_pred > 0, 1, -1)

strategy_returns = signal * y_test.values

# cumulative performance
cumulative = np.cumprod(1 + strategy_returns)

plt.figure()
plt.plot(cumulative)
plt.title("XGBoost Long/Short Strategy Equity Curve")
plt.show()