# XGBoost Regression Strategy (Nonlinear Alpha Model)

## Overview
This strategy improves upon the baseline by using a more powerful nonlinear model to predict next-day returns:

$$
X_{i,t} \rightarrow r_{i,t+1}
$$

It captures nonlinear relationships and interactions between features.

---

## Features

Same as baseline:
- return
- mom_5
- mom_20
- vol_20
- ma_gap

---

## Model

We use:

:contentReference[oaicite:0]{index=0}

Key idea:
- Sequential tree boosting
- Each tree corrects errors of previous ones
- Captures nonlinear patterns in financial data

---

## Trading Rule

- If predicted return > 0 → long
- If predicted return < 0 → short

---

## Intuition

Instead of simple averages, the model learns complex conditional rules like:
- momentum works only when volatility is low
- mean reversion depends on regime”

---

## Advantage over Random Forest

- Better handling of weak signals
- Captures nonlinear structure
- Typically more stable predictions

---

## Limitation

- Still predicts absolute returns
- Does not exploit cross-sectional relationships