# Google Trends + XGBoost Alpha Strategy (Explanation)

## Objective
The goal of this strategy is to predict short-term stock returns using a combination of:
- traditional market-based features (momentum, volatility, trend)
- alternative data (Google search interest as a proxy for investor attention)

We model the relationship:
$$
X_{i,t} \rightarrow r_{i,t+1}
$$
where features at time *t* are used to predict next-day returns.

---

## Features Used

For each stock and each day, we construct:

### 1. Price-based signals
- **return**: daily percentage change
- **mom_5**: 5-day momentum (short-term trend)
- **mom_20**: 20-day momentum (medium-term trend)
- **vol_20**: 20-day rolling volatility (risk / uncertainty proxy)
- **ma_gap**: deviation from 20-day moving average (mean reversion signal)

### 2. Alternative data signal (Google Trends)
- **trend**: normalized search interest for the ticker symbol

This captures **investor attention**, which often precedes price movement due to:
- retail trading activity
- news-driven hype
- information diffusion delays

---

## Intuition Behind the Strategy

The core hypothesis is:

> Markets are partially driven by attention, not just fundamentals.

When search interest in a stock increases:
- more investors are likely researching or reacting to news
- trading volume often increases
- short-term price pressure can emerge

The model learns nonlinear interactions between:
- price momentum (technical trend continuation)
- volatility (risk regime)
- attention spikes (Google Trends)

---

## Model (Machine Learning Layer)

We use **XGBoost regression**, which:
- captures nonlinear relationships
- handles feature interactions automatically
- is robust to noisy financial data

The model outputs:
$$
\hat{r}_{t+1}
$$
a prediction of next-day return.

---

## Trading Strategy (How We Invest)

At each time step:

### Step 1: Predict returns
- If predicted return > 0 → bullish view
- If predicted return < 0 → bearish view

### Step 2: Convert predictions into positions
- **Long (+1)** if model predicts positive return
- **Short (-1)** if model predicts negative return

### Step 3: Compute strategy return
\[
\text{strategy return}_t = \text{position}_t \times r_{t+1}
\]

### Step 4: Compound returns
We simulate portfolio growth over time:
$$
\text{equity}_t = \prod (1 + \text{strategy return}_t)
$$

---

## Evaluation Metrics

We evaluate performance using:

### 1. Prediction correlation
Measures alignment between:
- predicted returns
- actual returns

Higher = better directional accuracy.

### 2. Equity curve
Shows how a $1 initial investment grows over time using the strategy.

---

## Important Assumptions

- No transaction costs included
- No slippage or liquidity constraints
- Equal weight across assets
- Fully invested long/short exposure each period

---

## Key Insight

Even weak signals (like Google search interest) can improve prediction when combined with:
- momentum
- volatility structure
- nonlinear ML models

The edge in this strategy comes from **aggregating weak signals into a stronger predictive model** rather than relying on any single indicator.