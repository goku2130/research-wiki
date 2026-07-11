---
id: proceedings:actor-critic-algorithms
type: web
title: Actor-Critic Algorithms
url: https://proceedings.neurips.cc/paper/1786-actor-critic-algorithms.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

The paper "Actor-Critic Algorithms" by Konda and Tsitsiklis proposes and analyzes a class of actor-critic algorithms for the simulation-based optimization of Markov Decision Processes (MDPs) over a parameterized family of randomized stationary policies (RSPs).

**Core Problem:**
The core problem addressed is the optimization of a Markov Decision Process (MDP) to minimize an average cost function $\Lambda(\theta)$ over a parameterized family of randomized stationary policies $\Pi = \{\mu^\theta; \theta \in \mathbb{R}^n\}$. Existing methods, such as actor-only and critic-only approaches, have limitations: actor-only methods may suffer from high variance in gradient estimators and lack accumulation of past information, while critic-only methods, despite learning value functions, lack reliable guarantees for policy optimality. Actor-critic methods aim to combine the strengths of both but have been theoretically understood primarily for lookup table representations. This paper seeks to develop actor-critic algorithms for high-dimensional problems with convergence guarantees.

**Method/Recipe Step by Step:**

1.  **MDP and Policy Parameterization:**
    *   Consider an MDP with finite state space $S$ and finite action space $A$, and a cost function $g: S \times A \to \mathbb{R}$.
    *   Policies are randomized stationary policies $\mu^\theta(x, u)$, parameterized by $\theta \in \mathbb{R}^n$.
    *   Assumptions (A1) and (A2) are made regarding the differentiability of $\mu^\theta(x, u)$ and the ergodicity of the Markov chains induced by $\mu^\theta$.
    *   The average cost function is $\Lambda(\theta) = \sum_{x \in S, u \in A} g(x, u)\eta^\theta(x, u)$, where $\eta^\theta(x, u) = \pi^\theta(x)\mu^\theta(x, u)$ is the stationary probability of state-action pair $(x, u)$.
    *   The "differential" cost function $V^\theta(x)$ is defined by the Poisson equation: $\Lambda(\theta) + V^\theta(x) = \sum_{u \in A} \mu^\theta(x, u) [g(x, u) + \sum_{y} P_{xy}(u)V^\theta(y)]$.
    *   The q-function is defined as $q^\theta(x, u) = g(x, u) - \Lambda(\theta) + \sum_{y} P_{xy}(u)V^\theta(y)$.

2.  **Gradient of the Cost Function:**
    *   Theorem 1 states the gradient of the cost function: $\frac{\partial}{\partial \theta_i} \Lambda(\theta) = \sum_{x, u} \eta^\theta(x, u)q^\theta(x, u)\psi_i^\theta(x, u)$, where $\psi_i^\theta(x, u)$ is the $i$-th component of $\psi^\theta(x, u)$ such that $\nabla \mu^\theta(x, u) = \mu^\theta(x, u)\psi^\theta(x, u)$.
    *   This is re-expressed using an inner product $(\cdot, \cdot)_\theta$ defined as $(q_1, q_2)_\theta = \sum_{x, u} \eta^\theta(x, u)q_1(x, u)q_2(x, u)$.
    *   Thus, $\frac{\partial}{\partial \theta_i} \Lambda(\theta) = (q^\theta, \psi_i^\theta)_\theta$.

3.  **Critic's Role - Projection:**
    *   The key insight is that the critic does not need to learn the full $q^\theta$ function, but rather its projection onto a low-dimensional subspace $W_\theta = \text{span}\{\psi_1^\theta, \dots, \psi_n^\theta\}$.
    *   Let $\Pi_\theta: \mathbb{R}^{|S||A|} \to W_\theta$ be the projection operator. Then $(q^\theta, \psi_i^\theta)_\theta = (\Pi_\theta q^\theta, \psi_i^\theta)_\theta$.
    *   The critic's job is to compute an approximation of $\Pi_\theta q^\theta$. This is achieved using TD learning with a linear approximation architecture for the q-function: $Q_\phi^\theta(x, u) = \sum_{j=1}^m r_j \phi_j^\theta(x, u)$, where $r = (r_1, \dots, r_m) \in \mathbb{R}^m$ are the critic's parameters.
    *   The features $\phi_j^\theta$ are chosen such that their span $\Phi_\theta$ contains $W_\theta$. A straightforward choice is $m=n$ and $\phi_j^\theta = \psi_j^\theta$.

4.  **Two-Time-Scale Algorithm:**
    *   The actor and critic updates occur during a simulation of a single sample path of the controlled Markov chain.
    *   The actor parameters $\theta_k$ are updated on a slower time scale than the critic parameters $r_k$.
    *   **Critic Update (TD(1) and TD($\alpha$)):**
        *   The critic maintains parameters $r_k$, an estimate of the average cost $\Lambda_k$, and an eligibility trace $z_k$.
        *   Average cost update: $\Lambda_{k+1} = \Lambda_k + \gamma_k(g(X_k, U_k) - \Lambda_k)$.
        *   Critic parameter update: $r_{k+1} = r_k + \gamma_k(g(X_k, U_k) - \Lambda_k + Q_r^\theta(X_{k+1}, U_{k+1}) - Q_r^\theta(X_k, U_k))z_k$.
        *   **TD(1) Critic:** $z_{k+1} = \phi^\theta(X_{k+1}, U_{k+1})$ if $X_{k+1} \neq x^*$, else $z_{k+1} = \phi^\theta(X_{k+1}, U_{k+1})$ (where $x^*$ is a reference state).
        *   **TD($\alpha$) Critic:** $z_{k+1} = \alpha z_k + \phi^\theta(X_k, U_k)$.
    *   **Actor Update:**
        *   $\theta_{k+1} = \theta_k - \beta_k f(r_k) Q_r^\theta(X_{k+1}, U_{k+1}) \psi^\theta(X_{k+1}, U_{k+1})$.
        *   $\beta_k$ is a positive stepsize, and $f(r_k) > 0$ is a normalization factor satisfying assumptions (A3) and (A4).

**Key Formulas in LaTeX:**

*   Average Cost Function: $\Lambda(\theta) = \sum_{x \in S, u \in A} g(x, u)\eta^\theta(x, u)$
*   Poisson Equation for $V^\theta(x)$: $\Lambda(\theta) + V^\theta(x) = \sum_{u \in A} \mu^\theta(x, u) [g(x, u) + \sum_{y} P_{xy}(u)V^\theta(y)]$
*   Q-function: $q^\theta(x, u) = g(x, u) - \Lambda(\theta) + \sum_{y} P_{xy}(u)V^\theta(y)$
*   Gradient of Cost Function (Theorem 1): $\frac{\partial}{\partial \theta_i} \Lambda(\theta) = \sum_{x, u} \eta^\theta(x, u)q^\theta(x, u)\psi_i^\theta(x, u)$
*   Inner Product: $(q_1, q_2)_\theta = \sum_{x, u} \eta^\theta(x, u)q_1(x, u)q_2(x, u)$
*   Critic's Linear Approximation: $Q_r^\theta(x, u) = \sum_{j=1}^m r_j \phi_j^\theta(x, u)$
*   Average Cost Update: $\Lambda_{k+1} = \Lambda_k + \gamma_k(g(X_k, U_k) - \Lambda_k)$
*   Critic Parameter Update: $r_{k+1} = r_k + \gamma_k(g(X_k, U_k) - \Lambda_k + Q_r^\theta(X_{k+1}, U_{k+1}) - Q_r^\theta(X_k, U_k))z_k$
*   TD($\alpha$) Eligibility Trace: $z_{k+1} = \alpha z_k + \phi^\theta(X_k, U_k)$
*   Actor Parameter Update: $\theta_{k+1} = \theta_k - \beta_k f(r_k) Q_r^\theta(X_{k+1}, U_{k+1}) \psi^\theta(X_{k+1}, U_{k+1})$

**Key Quantitative Results and Numbers:**

*   **Convergence for TD(1) Critic (Theorem 2):**
    *   $\liminf_{k \to \infty} \|\nabla \Lambda(\theta_k)\| = 0$ with probability 1.
    *   If $\{\theta_k\}$ is bounded with probability 1, then $\lim_{k \to \infty} \|\nabla \Lambda(\theta_k)\| = 0$ with probability 1.
*   **Convergence for TD($\alpha$) Critic (Theorem 3):**
    *   For every $\epsilon > 0$, there exists $\alpha$ sufficiently close to 1 such that $\liminf_{k \to \infty} \|\nabla \Lambda(\theta_k)\| \le \epsilon$ with probability 1.
*   **Stepsize Assumptions (A6):** The stepsize sequences $\{\beta_k\}$ and $\{\gamma_k\}$ must be positive, non-increasing, and satisfy $\sum \beta_k = \infty$, $\sum \beta_k^2 < \infty$, $\sum \gamma_k = \infty$, $\sum \gamma_k^2 < \infty$. Additionally, $\beta_k / \gamma_k \to 0$, implying the actor updates on a slower time scale.

**Stated Limitations:**

*   The algorithms are gradient-based, so convergence to a globally optimal policy cannot be guaranteed; only convergence to a local minimum of $\Lambda(\theta)$ is expected.
*   For the TD($\alpha$) critic (where $\alpha < 1$), the critic generally converges to an approximation of the desired projection, leading to a weaker convergence guarantee where $\|\nabla \Lambda(\theta_k)\|$ only becomes small (infinitely often), rather than converging to zero.
*   The theoretical guarantees for the TD(1) critic appear stronger than for TD($\alpha$), although TD($\alpha$) is expected to perform better in practice due to smaller variance.
*   The analysis relies on assumptions such as the uniform positive definiteness of the matrix $G(\theta) = \sum_{x,u} \phi^\theta(x,u)\phi^\theta(x,u)^T$ (Assumption A5).
*   The algorithms are presented for finite state and action spaces, though the authors suggest extension to arbitrary spaces with ergodicity assumptions.
