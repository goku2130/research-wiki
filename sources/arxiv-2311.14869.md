---
id: arxiv:2311.14869
type: paper
title: On the Complexity of Computing Sparse Equilibria and Lower Bounds for No-Regret
  Learning in Games
url: https://arxiv.org/abs/2311.14869
retrieved: '2026-07-12'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Summary: On the Complexity of Computing Sparse Equilibria and Lower Bounds for No-Regret Learning in Games

## Core Problem
The authors investigate the fundamental computational complexity barriers of no-regret learning in games, specifically focusing on the time required for algorithms to converge to a **Coarse Correlated Equilibrium (CCE)**. While it is known that no-regret learners converge to a CCE, the authors seek to establish non-trivial, dimension-dependent lower bounds on the number of iterations required to reach an $\epsilon$-CCE in both normal-form and extensive-form games. The central technical vehicle for this is the study of **sparse CCE**, where a correlated distribution $\mu$ is $k$-sparse if it is a uniform mixture of $k$ product distributions:

$$
\boldsymbol{\mu}=\frac{1}{k}\sum_{\kappa=1}^{k}\bigotimes_{i=1}^{n}x_{i}^{(\kappa)}
$$

A 1-sparse CCE is equivalent to a Nash Equilibrium (NE).

## Method and Recipe
The authors establish their lower bounds via a reduction from the problem of computing a Nash Equilibrium in a two-player normal-form game $G$ to computing a sparse CCE in a constructed three-player extensive-form game $\mathcal{T}(G)$.

### Step-by-Step Reduction
1.  **Construction of $\mathcal{T}(G)$**: The authors create an extensive-form game $\mathcal{T}(G)$ consisting of $H$ repetitions of the original game $G$.
2.  **Introduction of the Kibitzer**: A third player, the "Kibitzer" (Player $K$), is added. The Kibitzer's action set $A_K$ consists of pairs $(i, a_i)$ where $i \in \{1, 2\}$ and $a_i \in A_i$.
3.  **Utility Definition**: Utilities in $\mathcal{T}(G)$ are defined such that the Kibitzer's reward depends on the difference between the actual action played by a player and the action suggested by the Kibitzer. This forces any CCE in $\mathcal{T}(G)$ to implicitly contain information about the NE of $G$.
4.  **Density Estimation**: To extract the NE of $G$ from a $T$-sparse $\epsilon$-CCE of $\mathcal{T}(G)$, the authors employ **Vovk’s aggregating algorithm** for online density estimation. This allows a deviating player to identify the product distribution used by other players.
5.  **Search for Equilibrium**: The algorithm searches through all states $s$ in the game tree $\mathcal{T}$ to find a pair of strategies $(\hat{q}_{1,s}, \hat{q}_{2,s})$ that constitutes a $(9\epsilon)$-NE of the original game $G$.

## Key Formulas
The performance of the learners is measured by **regret**, defined over a time horizon $T$ as:

$$
\mathsf{R e g}_{i}^{T}\triangleq\text{m a x}_{ {pm x}_{i}^{\star}\in\Delta(\mathcal{A}_{i})}\sum_{t=1}^{T}\langle{\pmb{x}}_{i}^{\star}-{\pmb{x}}_{i}^{(t)},{\pmb{u}}_{i}^{(t)}\rangle
$$

A distribution $\mu$ is an **$\epsilon$-CCE** if for any player $i$ and any deviation $a_i' \in A_i$:

$$
\mathbb{E}_{a\sim\mu}[u_{i}(\mathbf{a})]\geq\mathbb{E}_{a\sim\mu}[u_{i}(a_{i}^{\prime},\mathbf{a}_{-i})]-\epsilon
$$

## Key Quantitative Results
The results are predicated on the **Exponential Time Hypothesis (ETH) for PPAD**, which posits that no truly subexponential algorithms exist for PPAD-complete problems.

*   **Extensive-Form Games**: There is no polynomial-time algorithm (in the description of the game $\mathcal{T}$) that can compute a $2^{\log_2^{1/2-o(1)} |\mathcal{T}|}$-sparse $\epsilon$-CCE for a constant $\epsilon > 0$.
*   **No-Regret Learning (Extensive-Form)**: For any algorithm with polynomial iteration complexity, there exists a game $\mathcal{T}$ such that at least $T \geq 2^{\log_2^{1/2-o(1)} |\mathcal{T}|}$ repetitions are required for the average regret $\frac{1}{T} \max_i \mathsf{Reg}_i^T$ to fall below a constant $\epsilon$.
*   **Normal-Form Games (MWU/OMWU)**: For players using Multiplicative Weights Update (MWU) or Optimistic MWU (OMWU) in $m$-action games, at least $T \geq 2^{(\log_2 \log_2 m)^{1/2-o(1)}}$ repetitions are required to attain an $O(1)$-CCE.

## Stated Limitations
*   **Player Count**: The hardness results for extensive-form games apply to games with three or more players; the authors state it remains open whether these bounds extend to two-player games.
*   **Complexity Gap**: There remains a gap between the established lower bound ($2^{\log_2^{1/2-o(1)} |\mathcal{T}|}$) and the best-known upper bounds, which are nearly-linear in $|\mathcal{T}|$.
*   **Assumptions**: The results rely on the ETH for PPAD; the authors note that proving these bounds under the more plausible $P \neq PPAD$ conjecture would require a different approach because $\epsilon$ would need to be inversely polynomial in $m$.
