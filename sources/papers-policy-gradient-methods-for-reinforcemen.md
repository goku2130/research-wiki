---
id: papers:policy-gradient-methods-for-reinforcemen
type: web
title: Policy Gradient Methods for Reinforcement Learning with Function Approximation
url: https://papers.neurips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

## Core Problem

The paper addresses the theoretical intractability of establishing convergence guarantees for reinforcement learning (RL) algorithms that rely on approximating a value function and deriving a policy from it, especially when using function approximation. This "value-function approach" often leads to discontinuous policy changes and has been shown to fail to converge for simple MDPs and function approximators. The goal is to develop a method for RL with function approximation that *is* theoretically tractable and converges to a locally optimal policy.

## Method/Recipe Step by Step

The proposed approach is a policy gradient method where the policy is explicitly represented by its own function approximator, independent of the value function. The policy parameters are updated according to the gradient of expected reward.

1.  **Policy Representation:** Represent the policy $\pi(s, a, \theta)$ as a differentiable function approximator with parameters $\theta$. For example, a neural network where input is state, output is action probabilities, and weights are $\theta$.
2.  **Performance Metric:** Define a performance metric $\rho(\pi)$ for the policy. Two formulations are considered:
    *   **Average Reward:** $\rho(\pi) = \lim_{n \to \infty} \frac{1}{n} E\{r_1 + r_2 + \dots + r_n | \pi\}$, where $d^\pi(s)$ is the stationary distribution of states.
    *   **Start-State:** $\rho(\pi) = E\{\sum_{k=0}^\infty \gamma^k r_{k+1} | s_0, \pi\}$, where $\gamma \in [0,1]$ is a discount rate.
3.  **Policy Gradient Theorem (Theorem 1):** The gradient of the performance metric with respect to the policy parameters can be expressed as:
    $\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta} Q^\pi(s,a)$
    This theorem states that the gradient does not depend on the effect of policy changes on the state distribution, which simplifies gradient estimation via sampling.
4.  **Approximating $Q^\pi$:** Since $Q^\pi(s,a)$ is typically unknown, it is approximated by a learned function approximator $f_w(s,a)$ with parameters $w$.
5.  **Compatibility Condition (Theorem 2):** For the approximation to be useful in estimating the gradient, $f_w$ must satisfy two conditions:
    *   **Local Optimality of $f_w$:** The parameters $w$ are updated such that they converge to a local optimum where the error in $f_w$ is orthogonal to the gradient of the policy parameterization:
        $\sum_s d^\pi(s) \sum_a \pi(s,a) [Q^\pi(s,a) - f_w(s,a)] \frac{\partial f_w(s,a)}{\partial w} = 0$
    *   **Compatibility with Policy Parameterization:** The function approximator $f_w$ must be compatible with the policy parameterization in the sense that:
        $\frac{\partial f_w(s,a)}{\partial w} = \frac{\partial \pi(s,a)}{\partial \theta} \frac{1}{\pi(s,a)}$
        This condition implies that $f_w$ should be linear in the same features as the policy, normalized to be mean zero for each state. This suggests $f_w$ approximates the *advantage* function $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$, rather than $Q^\pi(s,a)$ directly.
6.  **Policy Gradient with Function Approximation (Theorem 2):** If $f_w$ satisfies the above conditions, then the gradient can be expressed as:
    $\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta} f_w(s,a)$
7.  **Policy Iteration with Function Approximation (Theorem 3):** A policy iteration algorithm is proposed where:
    *   The value function approximator $f_w$ is updated to satisfy the local optimality condition (step 5a).
    *   The policy parameters $\theta$ are updated in the direction of the gradient using a step size $\alpha_k$:
        $\theta_{k+1} = \theta_k + \alpha_k \sum_s d^{\pi_k}(s) \sum_a \frac{\partial \pi_k(s,a)}{\partial \theta} f_{w_k}(s,a)$
    This iterative process is proven to converge to a locally optimal policy under certain conditions.

## Key Formulas in LaTeX

*   **Policy Parameter Update:**
    $\Delta \theta \approx \alpha \frac{\partial \rho}{\partial \theta}$
*   **Average Reward Performance Metric:**
    $\rho(\pi) = \lim_{n \to \infty} \frac{1}{n} E\{r_1 + r_2 + \dots + r_n | \pi\} = \sum_s d^\pi(s) \sum_a \pi(s,a) \mathcal{R}_s^a$
*   **Start-State Performance Metric:**
    $\rho(\pi) = E\{\sum_{k=0}^\infty \gamma^k r_{k+1} | s_0, \pi\}$
*   **Policy Gradient Theorem (Theorem 1):**
    $\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta} Q^\pi(s,a)$
*   **Local Optimality Condition for $f_w$ (from Theorem 2):**
    $\sum_s d^\pi(s) \sum_a \pi(s,a) [Q^\pi(s,a) - f_w(s,a)] \frac{\partial f_w(s,a)}{\partial w} = 0$
*   **Compatibility Condition for $f_w$ (from Theorem 2):**
    $\frac{\partial f_w(s,a)}{\partial w} = \frac{\partial \pi(s,a)}{\partial \theta} \frac{1}{\pi(s,a)}$
*   **Policy Gradient with Function Approximation (Theorem 2):**
    $\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta} f_w(s,a)$
*   **Policy Iteration Update Rule for $\theta$ (Theorem 3):**
    $\theta_{k+1} = \theta_k + \alpha_k \sum_s d^{\pi_k}(s) \sum_a \frac{\partial \pi_k(s,a)}{\partial \theta} f_{w_k}(s,a)$

## Key Quantitative Results and Numbers

*   The paper proves for the first time that a version of policy iteration with arbitrary differentiable function approximation is convergent to a locally optimal policy.
*   The convergence proof relies on the conditions that the policy and value function approximators are differentiable, satisfy the compatibility condition (4), and have bounded partial derivatives (i.e., $\max_{\theta,s,a,i,j} |\frac{\partial^2 \pi(s,a)}{\partial \theta_i \partial \theta_j}| < B < \infty$).
*   The step-size sequence $\alpha_k$ must satisfy $\lim_{k \to \infty} \alpha_k = 0$ and $\sum_k \alpha_k = \infty$.

## Stated Limitations

*   The compatibility condition $\frac{\partial f_w(s,a)}{\partial w} = \frac{\partial \pi(s,a)}{\partial \theta} \frac{1}{\pi(s,a)}$ may only be satisfied if $f_w$ is linear in the features given on the right-hand side.
*   The convergence proof for policy iteration (Theorem 3) guarantees convergence to a *locally* optimal policy, not necessarily a globally optimal one.
*   The choice of an arbitrary function of state $v(s)$ added to the value function or its approximation (e.g., $f_w(s,a) + v(s)$) does not affect the theorems but can substantially affect the variance of the gradient estimators. In practice, $v(s)$ should presumably be set to the best available approximation of $V^\pi(s)$.
