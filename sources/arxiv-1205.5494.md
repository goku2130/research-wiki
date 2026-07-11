---
id: arxiv:1205.5494
type: paper
title: Improved Adaptive Rejection Metropolis Sampling Algorithms
url: https://arxiv.org/abs/1205.5494
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# Improved Adaptive Rejection Metropolis Sampling Algorithms

### Core Problem
Adaptive Rejection Metropolis Sampling (ARMS) is designed to sample from one-dimensional, non-log-concave target densities by combining adaptive rejection sampling (ARS) with a Metropolis-Hastings (MH) step. However, the authors identify a critical structural flaw: the adaptive mechanism only adds support points to the proposal density $\pi_t(x)$ when a sample is rejected by the rejection sampling (RS) test. This occurs only when $\pi_t(x) \geq p(x)$. Consequently, in regions where the proposal is below the target ($\pi_t(x) < p(x)$), support points may never be added. This prevents the sequence of proposals from converging to the target distribution, often causing the Markov chain to become trapped in a single mode of multimodal densities.

### Proposed Methods
The authors propose two variants, **$\text{A}^2\text{RMS}$** and **$\text{IA}^2\text{RMS}$**, which allow the incorporation of support points even when the proposal is below the target.

#### $\text{A}^2\text{RMS}$ Recipe
1. **Initialization**: Set $k=0, t=0$, choose initial value $x_0$, a stop-adaptation time $K$, and total iterations $N$.
2. **Proposal Construction**: Build a piecewise-linear approximation $W_t(x)$ of the potential function $V(x) = \log p(x)$ using a set of support points $\mathcal{S}_t$. The proposal is $\pi_t(x) = \exp(W_t(x))$.
3. **Sampling**: Draw $x' \sim \tilde{\pi}_t(x)$ and $u_1 \sim \mathcal{U}([0, 1])$.
4. **RS Control**: If $u_1 > p(x')/\pi_t(x')$, discard $x'$, add $x'$ to $\mathcal{S}_{t+1}$, update $t$, and return to Step 2.
5. **MH Control**: If $u_1 \leq p(x')/\pi_t(x')$, set $x_{k+1} = x'$ with probability $\alpha$, otherwise $x_{k+1} = x_k$.
6. **Additional Adaptation**: If $k \leq K$, draw $u_2 \sim \mathcal{U}([0, 1])$. If $u_2 > \pi_t(x'|\mathcal{S}_t)/p(x')$, add $x'$ to $\mathcal{S}_{t+1}$.
7. **Iteration**: Update $t = t+1, k = k+1$. Repeat until $k=N$.

#### $\text{IA}^2\text{RMS}$ Recipe
$\text{IA}^2\text{RMS}$ is an adaptive independent MH algorithm. It follows the $\text{A}^2\text{RMS}$ steps but introduces an auxiliary variable $y$ to ensure the proposal is independent of the current state $x_k$:
* If $x_{k+1} = x'$, then $y = x_k$.
* If $x_{k+1} = x_k$, then $y = x'$.
* The support set $\mathcal{S}_{t+1}$ is updated using $y$ instead of $x'$ in the second control step (Step 6 above).

### Key Formulas
The MH acceptance probability $\alpha$ is defined as:

$$
\alpha = \min\left[1, \frac{p(x') \min[p(x_k), \pi_t(x_k|\mathcal{S}_t)]}{p(x_k) \min[p(x'), \pi_t(x'|\mathcal{S}_t)]}\right]
$$

The discrepancy between the proposal and target is measured by the distance:

$$
D_{\pi|p}(t) = \int_{\mathcal{D}} |\pi_t(x) - p(x)| dx
$$

### Quantitative Results
The algorithms were tested on a mixture of three Gaussians with a true mean of $1.6$ over $N=5000$ iterations. Using the standard ARMS proposal construction (Procedure 1), the results averaged over 2000 runs were:

| Algorithm | Estimated Mean | Std of Estimation | Correlation | Avg. Support Points | Approx. $D_{\pi|p}(t)$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ARMS** | 1.6480 | 0.7301 | 0.3856 | 65.87 | 3.0020 |
| **$\text{A}^2\text{RMS}$** | 1.6244 | 0.1184 | 0.0038 | 94.50 | 0.0613 |
| **$\text{IA}^2\text{RMS}$** | 1.6233 | 0.1238 | 0.0041 | 94.84 | 0.0609 |

The proposed variants showed a significant reduction in estimation variance and linear correlation. Notably, $D_{\pi|p}(t)$ was more than one order of magnitude lower than that of standard ARMS.

### Stated Limitations
* **Convergence ($\text{A}^2\text{RMS}$)**: Because the current state $x_t$ can be incorporated into $\mathcal{S}_t$, the balance condition may be violated. To ensure convergence to the invariant density, the authors recommend setting $K < N$ and discarding the first $K$ samples.
* **Computational Cost**: There is a slight increase in computational overhead due to the additional control step and a moderate increase in the number of support points required to achieve convergence.
