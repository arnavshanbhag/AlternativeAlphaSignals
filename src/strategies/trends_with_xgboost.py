import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


trend_df = pd.read_csv("src/strategies/cache/google_trends_cache.csv")
trend_df["date"] = pd.to_datetime(trend_df["date"]).dt.normalize()
try:
    old = pd.read_csv("google_trends_cache.csv")
    old["date"] = pd.to_datetime(old["date"]).dt.normalize()
except FileNotFoundError:
    old = pd.DataFrame(columns=["date", "ticker", "trend"])


new = trend_df.copy()
new["date"] = pd.to_datetime(new["date"]).dt.normalize()

combined = pd.concat([old, new], ignore_index=True)

combined = combined.drop_duplicates(
    subset=["date", "ticker"],
    keep="last"
)


combined.to_csv("google_trends_cache.csv", index=False)

print("Cache updated. Total rows:", len(combined))

tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "TSLA", "LLY"]

prices = yf.download(tickers, start="2026-01-01")["Close"]

returns = prices.pct_change()
momentum_5 = prices.pct_change(5)
momentum_20 = prices.pct_change(20)
vol_20 = returns.rolling(20).std()
ma_20 = prices.rolling(20).mean()
ma_gap = prices / ma_20 - 1
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
    tmp["date"] = tmp.index

    df_list.append(tmp)

df = pd.concat(df_list).dropna()
df["date"] = pd.to_datetime(df["date"]).dt.normalize()

df = df.merge(trend_df, on=["date", "ticker"], how="left")

df["trend"] = df["trend"].fillna(0)


features = [
    "return",
    "mom_5",
    "mom_20",
    "vol_20",
    "ma_gap",
    "trend"
]

X = df[features]
y = df["target"]


split = int(len(df) * 0.8)

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

pred = model.predict(X_test)
print(pred)

signal = np.where(pred > 0.002, 1,
         np.where(pred < -0.002, -1, 0))
print(signal)
strategy_returns = signal * y_test.values

equity = (1 + strategy_returns).cumprod()
print(equity)

corr = np.corrcoef(pred, y_test)[0, 1]
print("Prediction correlation:", corr)

plt.figure()
plt.plot(equity)
plt.title("Cached Google Trends + XGBoost Strategy")
plt.show()

feature_cols = [
    "return",
    "mom_5",
    "mom_20",
    "vol_20",
    "ma_gap",
    "trend"
]

latest = df.sort_values("date").groupby("ticker").tail(1).copy()

X_live = latest[feature_cols]

latest["predicted_return_tomorrow"] = model.predict(X_live)


latest["position"] = np.where(latest["predicted_return_tomorrow"] > 0.005, 1,
         np.where(latest["predicted_return_tomorrow"] < -0.005, -1, 0))


latest["position_scaled"] = latest["predicted_return_tomorrow"] / (
    latest["vol_20"].replace(0, np.nan)
)

latest["position_scaled"] = latest["position_scaled"].fillna(0)

print("\nTOMORROW PREDICTIONS\n")
print(latest[[
    "ticker",
    "predicted_return_tomorrow",
    "position",
    "position_scaled"
]])

print(df["date"].max())
