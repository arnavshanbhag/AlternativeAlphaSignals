import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LOAD EXISTING CACHE (if it exists)
# --------------------------------------------------

trend_df = pd.read_csv("src/strategies/cache/build_cache.py")
trend_df["date"] = pd.to_datetime(trend_df["date"]).dt.normalize()
try:
    old = pd.read_csv("google_trends_cache.csv")
    old["date"] = pd.to_datetime(old["date"]).dt.normalize()
except FileNotFoundError:
    old = pd.DataFrame(columns=["date", "ticker", "trend"])

# --------------------------------------------------
# 2. NEW DATA (from your latest run)
#    assumes you already created trend_df
# --------------------------------------------------

new = trend_df.copy()
new["date"] = pd.to_datetime(new["date"]).dt.normalize()

# --------------------------------------------------
# 3. COMBINE + DEDUPLICATE
# --------------------------------------------------

combined = pd.concat([old, new], ignore_index=True)

combined = combined.drop_duplicates(
    subset=["date", "ticker"],
    keep="last"
)

# --------------------------------------------------
# 4. SAVE UPDATED CACHE
# --------------------------------------------------

combined.to_csv("google_trends_cache.csv", index=False)

print("Cache updated. Total rows:", len(combined))

# --------------------------------------------------
# 2. STOCK DATA
# --------------------------------------------------

tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "TSLA", "LLY"]

prices = yf.download(tickers, start="2026-01-01")["Close"]

returns = prices.pct_change()
momentum_5 = prices.pct_change(5)
momentum_20 = prices.pct_change(20)
vol_20 = returns.rolling(20).std()
ma_20 = prices.rolling(20).mean()
ma_gap = prices / ma_20 - 1
target = returns.shift(-1)

# --------------------------------------------------
# 3. BUILD FEATURE DATASET
# --------------------------------------------------

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

# --------------------------------------------------
# 4. MERGE CACHED TRENDS
# --------------------------------------------------

df = df.merge(trend_df, on=["date", "ticker"], how="left")

df["trend"] = df["trend"].fillna(0)

# --------------------------------------------------
# 5. FEATURES
# --------------------------------------------------

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

# --------------------------------------------------
# 6. TRAIN / TEST SPLIT
# --------------------------------------------------

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# --------------------------------------------------
# 7. XGBOOST MODEL
# --------------------------------------------------

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

# --------------------------------------------------
# 8. TRADING STRATEGY
# --------------------------------------------------
#signal = np.where(pred > 0, 1, -1)
signal = np.where(pred > 0.002, 1,
         np.where(pred < -0.002, -1, 0))
print(signal)
strategy_returns = signal * y_test.values

equity = (1 + strategy_returns).cumprod()
print(equity)

# --------------------------------------------------
# 9. EVALUATION
# --------------------------------------------------

corr = np.corrcoef(pred, y_test)[0, 1]
print("Prediction correlation:", corr)

plt.figure()
plt.plot(equity)
plt.title("Cached Google Trends + XGBoost Strategy")
plt.show()

# --------------------------------------------------
# 1. DEFINE FEATURE COLUMNS (must match training)
# --------------------------------------------------

feature_cols = [
    "return",
    "mom_5",
    "mom_20",
    "vol_20",
    "ma_gap",
    "trend"
]

# --------------------------------------------------
# 2. GET MOST RECENT DATA FOR EACH STOCK
# --------------------------------------------------

latest = df.sort_values("date").groupby("ticker").tail(1).copy()

# --------------------------------------------------
# 3. BUILD INPUT FEATURES FOR PREDICTION
# --------------------------------------------------

X_live = latest[feature_cols]

# --------------------------------------------------
# 4. PREDICT NEXT-DAY RETURNS
# --------------------------------------------------

latest["predicted_return_tomorrow"] = model.predict(X_live)

# --------------------------------------------------
# 5. CONVERT TO TRADING SIGNAL
#    (long if positive, short if negative)
# --------------------------------------------------

latest["position"] = np.where(latest["predicted_return_tomorrow"] > 0.002, 1,
         np.where(latest["predicted_return_tomorrow"] < -0.002, -1, 0))


# --------------------------------------------------
# 6. OPTIONAL: CONFIDENCE-BASED SIZING
# --------------------------------------------------

latest["position_scaled"] = latest["predicted_return_tomorrow"] / (
    latest["vol_20"].replace(0, np.nan)
)

latest["position_scaled"] = latest["position_scaled"].fillna(0)

# --------------------------------------------------
# 7. OUTPUT RESULTS
# --------------------------------------------------

print("\n📊 TOMORROW PREDICTIONS\n")
print(latest[[
    "ticker",
    "predicted_return_tomorrow",
    "position",
    "position_scaled"
]])

print(df["date"].max())
