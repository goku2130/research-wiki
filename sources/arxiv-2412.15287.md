---
id: arxiv:2412.15287
type: paper
title: Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models
url: https://arxiv.org/pdf/2412.15287
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# Inference-Aware Fine-Tuning for Best-of-N Sampling in LLMs

### Core Problem
Standard fine-tuning methods (SFT and RL) are typically agnostic to the inference-time strategy, training models to produce a single best response. However, in practice, Large Language Models (LLMs) often utilize inference-time compute via strategies like **Best-of-N (BoN)** sampling, where a verifier selects the best response from $N$ candidates. The authors argue that decoupling training from the inference strategy is suboptimal; if a model is intended for BoN sampling, it should be trained to optimize the exploration-exploitation trade-off—generating a diverse set of candidates that increase the probability of the verifier finding a correct solution.

### Method/Recipe
The authors propose an "inference-aware" paradigm that directly optimizes the performance of the BoN policy. Because the $\text{argmax}$ operator in BoN is non-differentiable, they derive variational and closed-form approximations to enable gradient-based optimization.

#### 1. BoN-Aware Supervised Fine-Tuning (BoN-SFT)
To optimize the likelihood of expert data under a BoN strategy, the authors use a variational approximation of the BoN policy:

$$
\pi_{\text{bon}}(y|x) \propto [\pi(y|x) \cdot \exp(\lambda_N Q_\pi(x, y))]
$$

where $Q_\pi(x, y) = \mathbb{E}_{y' \sim \pi(\cdot|x)}[\mathbf{1}_{r(x,y) \geqslant r(x,y')}]$ is the expected win-rate of response $y$ over the base policy $\pi$ according to verifier $r$, and $\lambda_N$ is a constant scaling with $N$. The gradient is computed by maximizing the likelihood of expert data while regularizing the policy to be more exploratory.

#### 2. BoN-Aware Reinforcement Learning (BoN-RL)
The goal is to maximize the environment reward $R(x, y)$ of the response selected by the BoN process:

$$
\max_{\pi \in \Pi} J(\pi) := \mathbb{E}_{x \sim P, y \sim \pi_{\text{bon}}(\cdot|x; \pi, r, N, T)}[R(x,y)]
$$

The authors derive a REINFORCE-style gradient that draws samples from the BoN distribution rather than the base policy $\pi$.

#### 3. BoN-RL for Binary Rewards (BoN-RLB)
For reasoning tasks with binary rewards ($R \in \{0, 1\}$), the authors derive a closed-form policy gradient that avoids value estimation. This method uses asymmetric weighting to prioritize "hard" examples (where the base failure probability $P_{\text{fail}}(x)$ is high):
*   **Positive weight:** $g_N^+(p) = \frac{N \cdot p^{N-1}}{1 - p^N}$
*   **Negative weight:** $g^-(p) = \frac{N \cdot p}{1 - p}$

A "positive-only" variant, **BoN-RLB(P)**, is also provided for scenarios where negative data is unavailable, using the weight $\bar{g}_N^+(p) := \frac{N \cdot p^{N-1} \cdot (1-p)}{(1-p^N)}$.

### Key Quantitative Results
Experiments using Gemma 2B and 9B models on mathematical and coding benchmarks demonstrate that inference-aware tuning outperforms standard SFT and RL:

*   **Hendrycks MATH (Gemma 2B):**
    *   **Bo32 Accuracy:** Improved from **26.8%** (base) to **30.8%** using BoN-RL-V.
    *   **pass@32:** Improved from **60.0%** (base) to **67.0%** using BoN-RL-S.
*   **HumanEval (Gemma 2B):**
    *   **pass@16:** Improved from **61.6%** (base) to **67.1%** using BoN-RLB(P).
*   **Generalization:** The models showed consistent improvements on held-out benchmarks (Functional MATH and MathOdyssey) and across various evaluation temperatures.

### Stated Limitations
*   **Verifier Noise:** At high temperatures ($T$) and large sample sizes ($N$), the verifier may suffer from Type II errors, mistakenly selecting "bad" random samples as the best responses due to misalignment with the true reward.
*   **Computational Cost:** Generating $N$ samples for every gradient update is computationally expensive. The authors suggest **BoN Distillation** (approximating the BoN distribution with a separate policy) to alleviate this.
*   **Training Instability:** For very hard problems ($P_{\text{fail}} \to 1$) and large $N$, the weight $g_N^+(p)$ can become extremely large, making training vulnerable to noise and requiring gradient clipping or regularization.
