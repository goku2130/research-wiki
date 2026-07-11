---
id: arxiv:2402.07319
type: paper
title: 'ODIN: Disentangled Reward Mitigates Hacking in RLHF'
url: https://arxiv.org/abs/2402.07319
retrieved: '2026-07-11'
maturity: comprehensive
topic: length-and-format-bias
---

# ODIN: Disentangled Reward Mitigates Hacking in RLHF

## Core Problem
A primary challenge in Reinforcement Learning from Human Feedback (RLHF) for Large Language Models (LLMs) is **reward hacking**, specifically in the form of **verbosity**. Because human raters often prefer longer, well-formatted responses regardless of actual helpfulness, Reward Models (RMs) learn a spurious correlation between response length and reward. Consequently, during RL fine-tuning, the policy exploits this vulnerability by generating excessively long responses to maximize reward without improving actual content quality.

## Method: Reward Disentanglement (ODIN)
To address this, the authors propose **ODIN**, a method that disentangles the reward signal into two separate components: a quality reward and a length reward.

### 1. Architecture
ODIN modifies the standard RM by replacing the single scalar output head with two linear projection heads ($W_Q$ and $W_L$) acting on shared feature representations. The total reward is the sum of the quality reward $r_\theta^Q(x, y)$ and the length reward $r_\theta^L(x, y)$.

### 2. Training Recipe
The RM is trained using a composite loss function to separate the signals:
1.  **Ranking Loss ($\mathcal{L}_R$):** A standard Bradley-Terry loss ensuring the combined reward of the chosen response ($y_w$) is higher than the rejected response ($y_l$):

$$
\mathcal{L}_R(x, y_w, y_l) = -\mathbb{E} [\log(\sigma((r_\theta^Q(x, y_w) + r_\theta^L(x, y_w)) - (r_\theta^Q(x, y_l) + r_\theta^L(x, y_l))))]
$$

2.  **Length Loss ($\mathcal{L}_L$):** This loss encourages $r_\theta^L$ to correlate with response length $L(y)$ and $r_\theta^Q$ to decorrelate from it:

$$
\mathcal{L}_L(x, y) = |\rho(r_\theta^L(x, y), L(y))| - \rho(r_\theta^Q(x, y), L(y))
$$

    where $\rho$ denotes the Pearson correlation computed within the global minibatch.
3.  **Orthogonality Loss ($\mathcal{L}_O$):** To ensure the two heads learn independent features, the authors enforce orthogonality between the projection weights:

$$
\mathcal{L}_O(\theta) = |W_Q W_L^T|
$$

**Total Loss:** $\mathcal{L} = \mathcal{L}_R + \lambda_L \mathcal{L}_L + \lambda_O \mathcal{L}_O$. Weight normalization is applied to $W_Q$ and $W_L$ to prevent the model from degenerating (e.g., setting $W_Q = 0$).

### 3. RL Fine-Tuning
During the RL stage (using PPO or ReMax), the length head $r_\theta^L$ is discarded. The policy is optimized using only the quality reward $r_\theta^Q$, effectively removing the incentive for length hacking.

## Key Quantitative Results
The authors established a "Pareto front" evaluation comparing **Win Score** (via GPT-4) against **Average Response Length**.

*   **Reward Model Correlation:** The baseline RM showed a strong Pearson correlation with length ($\rho = 0.451$). ODIN ($\lambda_L=1.0, \lambda_O=1.0$) reduced this correlation to $\rho = -0.03$.
*   **Accuracy:** ODIN maintained high preference prediction accuracy (69.2% validation accuracy) compared to the baseline RM (70.1%), despite removing the length signal.
*   **Policy Performance:** ODIN consistently achieved a higher Pareto front than baselines (PPO* and ReMax*), even when those baselines utilized reward clipping and length penalties.
*   **Benchmarks:** ODIN maintained or improved base capabilities compared to the SFT initialization. For example, on **TruthfulQA**, the score improved from 32.68 (SFT) to 34.64 (ODIN at length 230).

## Limitations
The authors note that unsupervised learning of disentangled representations is impossible without inductive biases on the models and data. Additionally, achieving perfect correlation and decorrelation is difficult in practice due to the use of minibatches. The current study specifically targets length hacking; the authors suggest that generalizing ODIN to other forms of reward hacking is a direction for future research.
