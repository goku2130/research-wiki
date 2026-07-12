---
id: arxiv:2109.08134
type: paper
title: Comparison and Unification of Three Regularization Methods in Batch Reinforcement
  Learning
url: https://arxiv.org/abs/2109.08134
retrieved: '2026-07-12'
maturity: comprehensive
topic: entropy-and-exploration
---

# Summary: Comparison and Unification of Three Regularization Methods in Batch Reinforcement Learning

## Core Problem
In batch reinforcement learning (RL), limited exploration of certain state-action pairs often leads to inaccurate model estimation and the learning of overly complex policies that overfit the available data. While various regularization methods exist to mitigate this, they typically operate on different components of the Markov Decision Process (MDP)—such as the discount factor, the transition matrix, or the policy set—making it difficult to compare them technically or intuitively.

## Unified Method
The authors propose a common framework that expresses three distinct regularization methods as a weighted average of the Maximum Likelihood Estimator (MLE) transition matrix $\hat{T}_{MLE}$ and a regularization matrix $T_{reg}$. The general form for the regularized transition matrix for action $a$ is:

$$
\hat{T}(a) = (1 - \epsilon)\hat{T}_{MLE}(a) + \epsilon T_{reg}(a)
$$

The three methods are unified as follows:

1.  **Bayesian Prior (Dirichlet):** The posterior mean transition matrix is a weighted average of the MLE and the prior mean transition matrix $T_{prior\_mean}$. The weight $\epsilon$ is defined by the prior parameters $\alpha_i$ and transition counts $c_i$:

$$
\epsilon = \frac{\sum \alpha_i}{\sum c_i + \sum \alpha_i}
$$

2.  **Discount Regularization:** Using a lower discount factor $\gamma_l$ (where $\gamma_l < \gamma$) is equivalent to replacing the transition matrix with a weighted average of the MLE and a matrix of zeros ($T_{zeros}$):

$$
\epsilon = \frac{\gamma - \gamma_l}{\gamma}
$$

3.  **Planning over $\epsilon$-greedy Policies:** Planning over stochastic $\epsilon$-greedy policies is equivalent to averaging the MLE transition matrix for action $a$ with the average of the MLE transition matrices for all possible actions $\mathcal{A}$:

$$
\hat{T}(a) = (1 - \epsilon)\hat{T}_{MLE}(a) + \epsilon \frac{1}{|\mathcal{A}|} \sum_{a'} \hat{T}_{MLE}(a')
$$

## Key Theoretical Insights
The framework reveals a connection between discount regularization and uniform priors. **Theorem 1** states that for two MDPs with identical state/action spaces and reward functions, one with discount factor $\gamma_l$ and one with discount factor $\gamma$ and a transition function $(1-\epsilon)T + \epsilon T_{unif}$ (where $T_{unif}$ is the uniform transition matrix), the optimal policies are identical.

The authors hypothesize that:
*   **Uniform priors** are most effective in "dense worlds" (highly interconnected states).
*   **Discount regularization** is best for balancing rewards across different timescales.
*   **$\epsilon$-greedy planning** is preferable for avoiding catastrophic outcomes by introducing stochasticity.

## Quantitative Results
The methods were evaluated across three MDPs: an **Interconnected Grid** (dense), **Cliff Walk** (catastrophic), and **Two Goals**.

*   **MDP Structure:** Under uniformly random data collection, the hypothesized interactions held: $\epsilon$-greedy planning performed best in the Cliff Walk, and uniform priors were effective in the Interconnected Grid.
*   **Data Collection Policy:** The choice of data collection policy significantly influenced results. When data was generated increasingly from the optimal policy, the relative performance of regularization methods shifted; in the "Two Goals" MDP, regularization ceased to be beneficial as the data became more optimal.
*   **MSE vs. Loss:** The authors found that a lower policy loss does not consistently correspond to a lower Mean Squared Error (MSE) of the estimated transition matrix.
*   **Data Size:** The number and length of trajectories had less impact on relative loss than the MDP structure and data collection policy.

## Stated Limitations
*   **Uniformity Assumptions:** The matrix form for the Bayesian prior requires assuming uniform visits across states and identical priors, which the authors admit is restrictive and unrealistic.
*   **Uneven Exploration:** The hypothesis that more flexible regularizers (like the Dirichlet prior) would outperform others under uneven exploration remained inconclusive.
*   **Formalization:** The authors note that further work is required to formally characterize the specific conditions under which each regularizer is preferred.
