---
id: arxiv:2509.23102
type: paper
title: Multiplayer Nash Preference Optimization
url: https://arxiv.org/abs/2509.23102
retrieved: '2026-07-11'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Multiplayer Nash Preference Optimization (MNPO)

Multiplayer Nash Preference Optimization (MNPO) is a framework that generalizes Nash Learning from Human Feedback (NLHF) from two-player interactions to $n$-player games.

### Core Problem
Standard Reinforcement Learning from Human Feedback (RLHF) based on the Bradley–Terry model often fails to capture the non-transitivity and heterogeneity of real-world human preferences. While recent NLHF methods (e.g., INPO, ONPO, EGPO) treat alignment as a two-player Nash game, they introduce a "single-opponent bias." This restriction fails to model the complexity of preference structures arising from multiple annotators, diverse evaluation criteria, or conflicting reward models, often leading to oscillatory behavior and narrow exploration.

### Method
MNPO formulates alignment as an $n$-player game where each policy competes against a population of opponents while being regularized toward a reference model $\pi_{\text{ref}}$.

#### 1. Homogeneous Setting (TD-MNPO)
In the homogeneous setting, all players share the same preference oracle. The authors propose **Time-dependent MNPO (TD-MNPO)**, where the opponent set is a weighted mixture of historical policies $\{\pi_{t-j}\}$.

**The Recipe:**
1. **Response Generation:** At iteration $t$, the current policy $\pi_t$ generates response pairs $(y^1, y^2)$.
2. **Preference Query:** A preference oracle $\mathbb{P}$ is queried to create a dataset $D_t$ of winning ($y_w$) and losing ($y_l$) responses.
3. **Policy Update:** The policy is updated by minimizing the TD-MNPO loss $\mathcal{L}_{\text{D}}^{t, \mathbb{D}}$, which measures the distance between the current log-odds margin and the weighted average log-odds margin of historical opponents.

#### 2. Heterogeneous Setting (HT-MNPO)
**Heterogeneous MNPO (HT-MNPO)** extends the framework to scenarios where each player $i$ has a distinct preference oracle $\mathbb{P}_i$ (e.g., different reward models for safety vs. helpfulness). Each policy $\pi_i$ optimizes its own objective $J_i$ against the population of other policies.

### Key Formulas
The general $n$-player objective for player $i$ is:

$$
J \left(\pi_{i}, \left\{\pi_{j} \right\}_{j \neq i}\right) = \mathbb {E} _{x \sim d_{0}} \left[ \mathbb {E} _{y^{i} \sim \pi_{i}, \left\{y^{j} \sim \pi_{j} \right\}_{j \neq i}} \left[ \mathbb {P} \left(y^{i} \succ \left\{y^{j} \right\}_{j \neq i} \mid x\right) \right] - \tau \text {K L} \left(\pi_{i} \| \pi_{\mathrm {ref}}\right) \right]
$$

The **TD-MNPO loss** is defined as:

$$
\mathcal{L} _{\text{D}}^{t,\mathbb{D}}(\pi\mid\beta,{\lambda_{j}},\eta)=\mathbb{E} _{y,y^{\prime}\sim\pi,y_{w},y_{l}\sim\lambda_{\mathbb{P}}(y,y^{\prime})}\mathbb{D}\left[\text{l o g}\frac{\pi(y_{w}\mid x)}{\pi(y_{l}\mid x)}-\sum_{j=0}^{n-2}\lambda_{j}\text{l o g}\frac{\pi_{t-j}(y_{w}\mid x)}{\pi_{t-j}(y_{l}\mid x)}\middle\| \eta\delta^{\star}\right]
$$

where $\mathbb{D}$ is a distance metric (e.g., squared distance), $\lambda_j$ are importance weights for historical policies, and $\delta^\star$ is the target reward gap.

The **unified duality gap** for player $i$ is:

$$
\text{D u a l G a p}(\pi)=\text{max} _{\pi^{\prime}\in\Pi}J(\pi^{\prime}, O_{\pi})-J(\pi, O_{\pi})
$$

where $O_\pi$ represents the fixed opponent policies. A Nash equilibrium is reached when $\text{DualGap}(\pi) = 0$.

### Key Quantitative Results
Evaluated using **Gemma-2-9B-it** as the base model and **GPT-5-mini** as the judge, TD-MNPO consistently outperformed baselines:

*   **Instruction Following:**
    *   **AlpacaEval 2.0 (LC WR):** TD-MNPO achieved **57.27%**, surpassing INPO (56.09%), SimPO (55.16%), and DPO (54.35%).
    *   **Arena-Hard (WR):** TD-MNPO achieved **52.26%**, compared to INPO's 48.03%.
    *   **MT-Bench:** TD-MNPO scored **7.03**, outperforming INPO (6.95%).
*   **Reasoning and Coding:**
    *   **AIME-24:** TD-MNPO was the only method to achieve a non-zero score (**3.33**); all other baselines and the SFT model scored 0.
    *   **HumanEval:** TD-MNPO achieved the best coding performance at **61.59**.
*   **Ablation on $n$:** Increasing the number of players from $n=1$ to $n=3$ improved the AlpacaEval 2.0 score from **53.32% to 57.27%**, with diminishing returns beyond $n=3$.

### Stated Limitations
The authors note that while TD-MNPO has provable convergence guarantees in homogeneous settings, **HT-MNPO lacks formal Nash equilibrium convergence guarantees**. This is because the heterogeneous setting creates a general-sum game that lacks the symmetry required for the multiplicative weights update convergence proofs. However, the authors state that HT-MNPO still yields effective empirical solutions.
