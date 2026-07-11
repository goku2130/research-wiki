---
id: arxiv:2603.17015
type: paper
title: Learning generalized Nash equilibria from pairwise preferences
url: https://arxiv.org/abs/2603.17015
retrieved: '2026-07-11'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Learning Generalized Nash Equilibria from Pairwise Preferences

### Core Problem
The authors address the challenge of finding a **Generalized Nash Equilibrium (GNE)** in a Generalized Nash Equilibrium Problem (GNEP) where the agents' objective functions $J_i$ are unknown. Traditional GNEP solvers typically require direct access to these objective functions or the ability to query agents for their best responses. The goal is to learn a GNE using only **pairwise preference queries** $\pi_i(x_i^1, x_i^2; x_{-i})$, which indicate whether agent $i$ prefers decision $x_i^1$ over $x_i^2$ given the decisions of other agents $x_{-i}$.

### Method
The proposed method employs an Active Learning (AL) loop to train surrogate objective functions $\hat{J}_i$ parameterized by $\theta_i$. The equilibrium of the learned surrogate GNEP is used to approximate the GNE of the underlying unknown problem.

#### 1. Preference Learning
The authors pose the learning of $\theta_i$ as a logistic regression classification problem. The probability that agent $i$ prefers $x_i^1$ over $x_i^2$ is modeled using a sigmoid function:

$$
P_{i}(x_{i}^{1},x_{i}^{2},x_{-i})=\frac{1}{1+\exp\left(\frac{\hat{J}_{i}^{1}-\hat{J}_{i}^{2}}{d_{i}(x_{i}^{1},x_{i}^{2})}\right)}
$$

where $\hat{J}_i^1$ and $\hat{J}_i^2$ are the surrogate values for the two decisions. To improve accuracy when decisions are very close, a dissimilarity function $d_i$ is introduced:

$$
d_{i}(x_{i}^{1},x_{i}^{2})=\log\bigl(\|x_{i}^{1}-x_{i}^{2}\|_{\infty}+1+\epsilon_{d}\bigr)
$$

The parameters $\theta_i$ are learned by minimizing a regularized cross-entropy loss:

$$
\min_{\theta_{i}\in\Theta_{i}} r_{i}(\theta_{i})+\frac{1}{M}\sum_{j=1}^{M}\mathcal{L}(\pi_{i}^{j},P_{i}(x_{i}^{j,1},x_{i}^{j,2},x_{-i}^{j}))
$$

where $r_i(\theta_i) = \rho_i \|\theta_i\|_2^2$ is an $\ell_2$ penalty and $\mathcal{L}$ is the cross-entropy loss.

#### 2. Active Learning Loop
The AL strategy balances exploration of the decision space and exploitation of the learned surrogate GNEP:
1. **Exploration:** A function $z_i^k(x_i)$ (e.g., a concave quadratic $z_i^k(x_i) = -\frac{1}{2}\|x_i - \bar{x}_i^k\|_2^2$) is used to promote space-filling.
2. **Query Point Selection:** A new query point $x^k$ is obtained by solving a GNEP that combines the surrogate function and the exploration term:

$$
x_{i}^{k}(x_{-i}^{k})\in\arg\min_{x_{i}\in\mathcal{X}_{i}}\hat{J}_{i}(x_{i},x_{-i}^{k};\theta_{i}^{k-1})-\delta^{k}z_{i}^{k}(x_{i})
$$

3. **Pairwise Comparison:** The agent is queried for their preference between $x_i^{k,1}$ (the solution above) and $x_i^{k,2}$, which is a noise-altered best response:

$$
x_{i}^{k,2}=\hat{x}_{i}^{k,2}+\sigma^{k}w^{k}\|\hat{x}_{i}^{k,2}\|_{\infty}
$$

   where $w^k$ is sampled from a uniform distribution $[-0.5, 0.5]$.
4. **Parameter Decay:** The weighting factors $\delta^k$ and $\sigma^k$ decay exponentially to shift from exploration to exploitation:

$$
\delta^{k}=\delta\left(1-\frac{k}{k_{\max}}\right)^{p_{\delta}}, \quad \sigma^{k}=\sigma\left(1-\frac{k}{k_{\max}}\right)^{p_{\sigma}}
$$

### Key Quantitative Results
The method was tested on game-theoretic Linear Quadratic Regulator (LQR) problems and literature GNEPs. For LQR problems, performance was measured using normalized Root Mean Square Error (RMSE) of closed-loop costs and the maximum best-response deviation $\max_i J_i$.

**Table I: LQR Results (at $k_{\max}=200$)**
| System Dimensions ($n_\xi=m, N$) | RMSE | $\max_i J_i$ |
| :--- | :--- | :--- |
| $n_\xi=m=6, N=3$ | 0.00141 | 0.0084 |
| $n_\xi=m=8, N=3$ | 0.00490 | 0.0391 |
| $n_\xi=m=12, N=4$ | 0.00368 | 0.1799 |

Hyperparameters used included $\sigma=0.3$, $\underline{\delta}=\underline{\sigma}=0.001$, $p_\delta=5$, $p_\sigma=4$, and an initial dataset size $M_0=50$.

### Stated Limitations
* **Problem Dependency:** The initial exploration weight $\delta$ is problem-dependent, as it must be scaled according to the decision variables and constraints.
* **Over-fitting:** Over-parameterizing the surrogate functions $\hat{J}_i$ can lead to near-perfect fits of preferences (over-fitting), which results in poor GNE approximation performance.
* **Constraint Knowledge:** The method assumes that local constraints $\mathcal{X}_i$ and global constraints $g(x) \leq 0, h(x) = 0$ are known.
