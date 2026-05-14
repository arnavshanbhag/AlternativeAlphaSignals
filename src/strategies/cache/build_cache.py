from pytrends.request import TrendReq
import pandas as pd
import time

tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", "LLY", "LAC", "NVDA"]

pytrends = TrendReq(hl="en-US", tz=360)

trend_frames = []

for t in tickers:
    try:
        pytrends.build_payload([t], timeframe="today 12-m")
        df = pytrends.interest_over_time()

        if df.empty:
            continue

        df = df[[t]].rename(columns={t: "trend"})
        df["ticker"] = t
        df = df.reset_index()

        trend_frames.append(df)

        print(f"Downloaded {t}")
        time.sleep(2)

    except Exception as e:
        print(f"Error {t}: {e}")

trend_df = pd.concat(trend_frames, ignore_index=True)

trend_df.to_csv("google_trends_cache.csv", index=False)

print("Saved cache.")