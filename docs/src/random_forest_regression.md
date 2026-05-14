# Random Forest Regression Strategy (Baseline Alpha Model)

## Overview
This strategy attempts to predict **next-day stock returns directly** using simple technical indicators. It is a supervised regression problem:

$$
X_{i,t} \rightarrow r_{i,t+1}
$$

Each stock is treated independently, and predictions are used to generate trading signals.

---

## Features (Inputs)

We use standard technical indicators:

- **return**: daily return \( r_{i,t} \)
- **mom_5**: 5-day momentum (short-term trend)
- **mom_20**: 20-day momentum (medium-term trend)
- **vol_20**: 20-day rolling volatility (risk proxy)
- **ma_gap**: deviation from moving average (mean reversion signal)

---

## Target

- **target = next-day return**
\[
r_{i,t+1}
\]

---

## Model

We use:
- Random Forest Regressor

This model captures nonlinear interactions between features.

---

## Trading Rule

- If predicted return > 0 → go **long**
- If predicted return < 0 → go **short**

---

## Strategy Return

$$
R_t = \text{signal}_t \cdot r_{t+1}
$$

---

## Intuition

Buy stocks the model expects to go up tomorrow and short those expected to go down

---

## Limitation

- No cross-sectional comparison between stocks
- Highly noisy at daily frequency
- Weak predictive signal in isolation