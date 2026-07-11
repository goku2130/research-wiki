---
id: arxiv:2409.03915
type: paper
title: Asynchronous Stochastic Approximation with Applications to Average-Reward Reinforcement
  Learning
url: https://arxiv.org/abs/2409.03915
retrieved: '2026-07-11'
maturity: comprehensive
topic: async-and-off-policy-rl
---

# Summary: Asynchronous Stochastic Approximation with Applications to Average-Reward Reinforcement Learning

## Core Problem
The paper addresses the stability and convergence of asynchronous stochastic approximation (SA) algorithms, specifically targeting average-reward reinforcement learning (RL) for Markov Decision Processes (MDPs) and semi-Markov Decision Processes (SMDPs). In average-reward settings, the underlying mappings typically lack the contraction or nonexpansion properties found in discounted reward settings, rendering existing asynchronous SA theory inadequate. The authors seek to establish general stability and convergence guarantees under broad noise conditions—including biased noise arising from estimated holding times in SMDPs—to provide a theoretical foundation for relative value iteration (RVI)-based RL algorithms.

## Method and Recipe
The authors employ a dynamical systems approach, relating the discrete SA iterates to the trajectories of an associated ordinary differential equation (ODE).

### 1. Algorithmic Framework
The algorithm operates in $\mathbb{R}^d$. At each iteration $n$, a nonempty subset $Y_n \subset \{1, \dots, d\}$ is selected. The components are updated as:

$$
x_{n+1}(i) = x_n(i) + \alpha_{\nu(n,i)}(h_i(x_n) + M_{n+1}(i) + \epsilon_{n+1}(i)), \quad \text{if } i \in Y_n
$$

where $\nu(n,i)$ is the cumulative number of updates to component $i$, $\alpha$ is a diminishing stepsize, $h$ is a Lipschitz continuous function, $M_{n+1}$ is a martingale difference sequence, and $\epsilon_{n+1}$ is a biased noise term.

### 2. Stability Analysis
To prove the boundedness of iterates $\{x_n\}$, the authors extend the Borkar-Meyn stability criterion. Because the noise terms in RL applications may not be bounded by deterministic constants, the authors use **stopping-time techniques** to construct auxiliary processes $\tilde{x}^n(\cdot)$ that are "better-behaved." These auxiliary processes are shown to track the solutions of scaled ODEs $\dot{x} = \lambda(t)h_c(x)$, where $h_c(x) = h(cx)/c$.

### 3. Convergence and Shadowing
Once stability is established, the authors use the concept of **shadowing** from dynamical systems to sharpen convergence results. They analyze whether the interpolated trajectory of the SA algorithm asymptotically coincides with a unique solution of the limiting ODE $\dot{x} = h(x)$. This involves decomposing the tracking error into stochastic noise and asynchrony components.

## Key Formulas
*   **Biased Noise Condition:** $\|\epsilon_{n+1}\| \leq \delta_{n+1}(1 + \|x_n\|)$, where $\delta_{n+1} \to 0$ almost surely (a.s.) as $n \to \infty$.
*   **Martingale Variance Bound:** $\mathbb{E}[\|M_{n+1}\|^2 \mid \mathcal{F}_n] \leq K_n(1 + \|x_n\|^2)$ a.s., with $\sup_n K_n < \infty$ a.s.
*   **Step-size Classes:**
    *   Class-1: $\alpha_n = \frac{1}{An}$
    *   Class-2: $\alpha_n = \frac{1}{An \ln n}$
*   **Limiting ODE:** $\dot{x}(t) = h(x(t))$, with equilibrium set $E_h := \{x \in \mathbb{R}^d \mid h(x) = 0\}$.

## Key Quantitative Results
*   **Theorem 2.1 (Stability):** Under the specified assumptions on $h$, noise, stepsizes, and asynchrony, the sequence $\{x_n\}$ is bounded a.s.
*   **Theorem 2.2 (General Convergence):** The sequence $\{x_n\}$ converges a.s. to a compact, connected, internally chain transitive, invariant set $D$ of the ODE $\dot{x} = h(x)$.
*   **Theorem 2.3 (Unique Convergence):** If $E_h$ is globally asymptotically stable and every ODE solution converges to a unique point in $E_h$, then $\{x_n\}$ converges a.s. to a unique (sample path-dependent) point in $E_h$ provided:
    *   For Class-1 stepsizes: $A > L_h$ (where $L_h$ is the Lipschitz constant of $h$) and the asynchrony condition $\gamma A > L_h$ holds.
    *   For Class-2 stepsizes: $A > L_h$.
    *   Noise condition: $\mu_\delta < -L_h$, where $\mu_\delta$ is the limit superior of $\frac{\ln(\delta_{n+1})}{\sum_{k=0}^n \alpha_k}$.

## Stated Limitations
The current analysis is restricted to finite-dimensional spaces $\mathbb{R}^d$. Additionally, the authors note that their framework does not currently account for communication delays in distributed computing environments, identifying this as a primary direction for future research.
