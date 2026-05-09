# AlternativeAlphaSignals
A repository dedicated to generating significant alpha signals from alternative data such as Reddit posts, foot traffic, X posts, weather, etc.

We define $r_{i,t+1}$ as the future return of asset $i$. Let $X_{i,t}$ be an alpha signal. We say that such a signal is significant if 
$$\mathbb E[r_{i,t+1}] \neq \mathbb E[r_{i,t+1} ~|~ X_{i,t}].$$

Our goal is to generate significant alpha signals from alternative data, such as reddit posts, foot traffic, X, etc.

We will train models on old data then apply our model to new data and track our profits/losses for different 
trading strategies.

