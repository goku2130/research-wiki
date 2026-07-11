---
id: arxiv:2107.11820
type: paper
title: A survey of Monte Carlo methods for parameter estimation
url: https://arxiv.org/abs/2107.11820
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# Summary of Monte Carlo Methods for Parameter Estimation

### Core Problem
In statistical signal processing, the objective is to estimate a set of static parameters $\boldsymbol{\theta}$ given observed data $\mathbf{y}$. This typically requires solving multi-variate optimization problems (e.g., Maximum Likelihood or Maximum A Posteriori estimators) or performing multi-dimensional integration (e.g., Minimum Mean Squared Error estimators). Because analytical closed-form expressions for these estimators are usually unavailable in real-world applications, Monte Carlo (MC) methods are employed to approximate these integrals using stochastic simulation.

### Bayesian Framework and Basic MC Method
The problem is formulated using the posterior probability density function (PDF), $\bar{\pi}(\boldsymbol{\theta}|\mathbf{y})$, derived via Bayes' rule:

$$
\bar{\pi}(\boldsymbol{\theta}|\mathbf{y}) = \frac{\ell(\mathbf{y}|\boldsymbol{\theta})p_0(\boldsymbol{\theta})}{Z(\mathbf{y})} = \frac{\pi(\boldsymbol{\theta}|\mathbf{y})}{Z(\mathbf{y})}
$$

where $\ell(\mathbf{y}|\boldsymbol{\theta})$ is the likelihood, $p_0(\boldsymbol{\theta})$ is the prior, and $Z(\mathbf{y})$ is the marginal likelihood. The goal is to compute the expectation of an integrable function $g(\boldsymbol{\theta})$:

$$
I = \mathbb{E}_{\bar{\pi}}(g(\boldsymbol{\theta})) = \int_{\Theta} g(\boldsymbol{\theta})\bar{\pi}(\boldsymbol{\theta}|\mathbf{y})d\boldsymbol{\theta}
$$

The basic MC method approximates this integral by drawing $M$ independent and identically distributed (IID) samples $\boldsymbol{\theta}^{(m)} \sim \bar{\pi}(\boldsymbol{\theta}|\mathbf{y})$ and calculating the unweighted sample average:

$$
\widehat{I}_M = \frac{1}{M} \sum_{m=1}^M g(\boldsymbol{\theta}^{(m)})
$$

### Methodologies

#### 1. Rejection Sampling (RS)
RS generates samples from a target $\bar{\pi}(\boldsymbol{\theta})$ using a simpler proposal $\bar{q}(\boldsymbol{\theta})$.
1. Draw $\boldsymbol{\theta}^{(t)} \sim \bar{q}(\boldsymbol{\theta})$ and $u \sim \mathcal{U}([0, 1))$.
2. Accept $\boldsymbol{\theta}^{(t)}$ if $u \le \frac{\pi(\boldsymbol{\theta}^{(t)})}{Cq(\boldsymbol{\theta}^{(t)})}$, where $C$ is a constant such that $Cq(\boldsymbol{\theta}) \ge \pi(\boldsymbol{\theta})$ for all $\boldsymbol{\theta} \in \Theta$.

#### 2. Markov Chain Monte Carlo (MCMC)
MCMC creates an ergodic Markov chain whose stationary distribution is the target PDF.

**Metropolis-Hastings (MH) Algorithm:**
1. Draw a candidate $\boldsymbol{\theta}' \sim q(\boldsymbol{\theta}|\boldsymbol{\theta}^{(t-1)})$.
2. Compute the acceptance probability:

$$
\alpha_t = \min \left[ 1, \frac{\pi(\boldsymbol{\theta}')q(\boldsymbol{\theta}^{(t-1)}|\boldsymbol{\theta}')}{\pi(\boldsymbol{\theta}^{(t-1)})q(\boldsymbol{\theta}'|\boldsymbol{\theta}^{(t-1)})} \right]
$$

3. Accept $\boldsymbol{\theta}'$ if $u \le \alpha_t$; otherwise, set $\boldsymbol{\theta}^{(t)} = \boldsymbol{\theta}^{(t-1)}$.
4. Discard initial samples (burn-in period $T_b$) and compute the average over the remaining $T-T_b$ samples.

**Gibbs Sampler:**
The Gibbs sampler updates each coordinate $\theta_{d_t}$ by drawing from its full conditional PDF:

$$
\theta_{d_t}^{(t)} \sim \bar{\pi}(\theta_{d_t} | \boldsymbol{\theta}_{\neg d_t}^{(t-1)})
$$

Coordinates are selected via **Systematic Scan** (fixed order), **Symmetric Scan** (ascending then descending), or **Random Scan** (uniform random selection).

**MH-within-Gibbs:**
When full conditional PDFs are non-standard and cannot be sampled directly, an internal MH algorithm is used to sample each coordinate. This hybrid approach performs $T_{MH}$ internal MH iterations per Gibbs update.

### Key Quantitative Results
*   **Optimal Acceptance Rates (AR):** For the random walk MH algorithm, the optimal AR is approximately $44\%$ for $D_\theta = 1$ and declines to $23.4\%$ as $D_\theta \to \infty$. A general practical rule of thumb is to tune the proposal variance to maintain an AR between $25\%$ and $40\%$.
*   **Convergence:** The MC estimate $\widehat{I}_M$ is unbiased and converges almost surely to $I$ as $M \to \infty$ per the strong law of large numbers.

### Stated Limitations
*   **Rejection Sampling:** Finding the bounding constant $C$ is difficult in high-dimensional spaces. If $Cq(\boldsymbol{\theta})$ is much larger than $\pi(\boldsymbol{\theta})$, the acceptance probability $P_A = Z_\pi / CZ_q$ becomes very low, rendering the method inefficient.
*   **MCMC:** Convergence speed depends heavily on the proposal PDF. If the chain is not **geometrically ergodic**, it may suffer from "heavy-tailed excursions," leading to instability and inaccurate estimation.
*   **Gibbs Sampling:** Requires the ability to sample from full conditional PDFs, which may not be possible for all models.
