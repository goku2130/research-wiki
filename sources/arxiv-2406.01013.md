---
id: arxiv:2406.01013
type: paper
title: Scalable Ensembling For Mitigating Reward Overoptimisation
url: https://arxiv.org/abs/2406.01013
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-model-overoptimization
---

# Summary: Scalable Ensembling For Mitigating Reward Overoptimisation

### Core Problem
In Reinforcement Learning from Human Feedback (RLHF), policies often suffer from **reward over-optimization**. This occurs when a policy overfits a learned "proxy" reward model, leading to a performance inflection point where the reward credited by a more performant "gold" reward model begins to degrade despite the proxy reward increasing. While computing a pessimistic statistic over an ensemble of reward models can mitigate this, maintaining multiple full-scale reward models is computationally prohibitive for large language models (LLMs) due to extreme memory and time requirements.

### Method
The authors propose a **multi-head reward model** that utilizes a shared encoder (backbone) with multiple independent linear heads. This architecture aims to provide the diversity of an ensemble while significantly reducing the computational overhead.

**Step-by-Step Recipe:**
1.  **Supervised Fine-Tuning (SFT):** A base model is fine-tuned to minimize perplexity on a set of instructions (e.g., 52k samples from the Alpaca dataset).
2.  **Multi-Head Reward Learning:** 
    *   The SFT model is used as a fixed feature extractor $F$.
    *   Multiple linear reward heads $H_i$ are initialized.
    *   These heads are fine-tuned using the Bradley-Terry loss on pairwise preference labels (preferred vs. dispreferred completions).
3.  **PPO Integration:** The SFT policy is further tuned using Proximal Policy Optimization (PPO). To mitigate over-optimization, the reward used for the policy update is the **minimum** reward predicted across all linear heads.

### Key Formulas
The PPO objective is defined as:

$$
\max_{\pi} \mathbb{E}_{s,a \sim \pi_{old}} [R(s,a) - \beta KL(\pi_{old} || \pi)]
$$

The reward for a specific head $i$ given input $x$ is:

$$
r_i(x) = H_i(F(x))
$$

The final pessimistic reward $\hat{r}(x)$ is the minimum across the ensemble of $i$ heads:

$$
\hat{r}(x) = \min_i H_i(F(x))
$$

### Quantitative Results and Numbers
The researchers evaluated the method using the **Alpaca Instructions** dataset via the AlpacaFarm framework, employing the **OPT model family** (1.3B parameter model as the proxy/policy and 6.7B as the gold model).

*   **Performance:** The multi-head reward model (with 3 heads) achieved "gold" reward performance similar to that of a full ensemble of three separate reward models, while significantly outperforming standard PPO, which exhibited the characteristic "concave-down" degradation of gold rewards.
*   **Training Efficiency:** The multi-head approach required only **one epoch** of reward training to be effective, whereas the full ensemble required at least **three epochs** to prevent over-optimization.
*   **Scalability:** The authors noted that while they fixed the ensemble size at three (as prior work found diminishing returns at four or five), the multi-head approach is amenable to much larger ensembles with minimal additional overhead compared to full ensembles.
*   **PPO Duration:** PPO was run for **15 epochs** to extensively test for over-optimization.

### Stated Limitations
*   **Head Count Threshold:** The authors suggest that increasing the number of heads beyond a certain threshold (e.g., 3) might introduce more noise than beneficial diversity, potentially leading to overfitting on dataset nuances.
*   **Calibration:** The minimum reward estimate is noted to be "technically miscalibrated." However, the authors argue this creates a regularization effect that prevents the model from reaching the local minima typical of over-optimized models.
*   **Scope:** The experiments were limited to the AlpacaFarm dataset and OPT models; the authors suggest further investigation into larger model sizes and other datasets (e.g., Stanford Human Preferences).
