---
id: arxiv:1707.06347
type: paper
title: Proximal Policy Optimization Algorithms
url: https://arxiv.org/pdf/1707.06347
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

The paper "Proximal Policy Optimization Algorithms" introduces a new family of policy gradient methods called Proximal Policy Optimization (PPO) for reinforcement learning.

**Core Problem:**
Existing deep reinforcement learning methods face challenges in scalability, data efficiency, and robustness. Q-learning struggles with continuous control, vanilla policy gradient methods are data-inefficient and lack robustness, and Trust Region Policy Optimization (TRPO) is complex to implement and incompatible with certain neural network architectures (e.g., those with dropout or parameter sharing). The goal is to develop an algorithm that combines the data efficiency and reliable performance of TRPO with the simplicity and generality of first-order optimization methods.

**Method/Recipe Step by Step:**
PPO algorithms alternate between sampling data from the environment and optimizing a "surrogate" objective function using stochastic gradient ascent. Unlike standard policy gradient methods that perform one gradient update per data sample, PPO allows for multiple epochs of minibatch updates on the same sampled data.

The core of PPO lies in its novel objective functions:

1.  **Clipped Surrogate Objective ($L^{CLIP}$):**
    *   Define the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)}$, where $\theta_{\mathrm{old}}$ are the policy parameters before the update.
    *   The objective is designed to penalize policy changes that move $r_t(\theta)$ too far from 1.
    *   The objective function is:
        $L ^ {C L I P} (\theta) = \hat {\mathbb {E}} _ {t} \left[ \min (r _ {t} (\theta) \hat {A} _ {t}, \text{clip} (r _ {t} (\theta), 1 - \epsilon , 1 + \epsilon) \hat {A} _ {t}) \right]$
        where $\epsilon$ is a hyperparameter (e.g., 0.2). This objective takes the minimum of the unclipped surrogate objective term ($r_t(\theta) \hat{A}_t$) and a clipped version of it. The clipping limits the probability ratio $r_t(\theta)$ to the interval $[1-\epsilon, 1+\epsilon]$, effectively creating a pessimistic estimate (lower bound) of the policy's performance.

2.  **Adaptive KL Penalty Coefficient ($L^{KLPEN}$):**
    *   This is an alternative or additive approach that uses a KL divergence penalty, where the penalty coefficient $\beta$ is adapted to achieve a target KL divergence $d_{targ}$ for each policy update.
    *   The objective function is:
        $L ^ {K L P E N} (\theta) = \hat {\mathbb {E}} _ {t} \left[ \frac {\pi_ {\theta} (a _ {t} \mid s _ {t})}{\pi_ {\theta_ {\mathrm{old}}} (a _ {t} \mid s _ {t})} \hat {A} _ {t} - \beta \text{KL} [ \pi_ {\theta_ {\mathrm{old}}} (\cdot \mid s _ {t}), \pi_ {\theta} (\cdot \mid s _ {t}) ] \right]$
    *   The adaptation rule for $\beta$ is:
        *   Compute $d = \hat{\mathbb{E}}_t[\mathrm{KL}[\pi_{\theta_{\mathrm{old}}}(\cdot |s_t),\pi_\theta (\cdot |s_t)]]$.
        *   If $d < d_{\mathrm{targ}} / 1.5$, then $\beta \leftarrow \beta / 2$.
        *   If $d > d_{\mathrm{targ}} \times 1.5$, then $\beta \leftarrow \beta \times 2$.

3.  **Combined Objective (Actor-Critic Style):**
    *   For architectures sharing parameters between policy and value function, the objective combines the policy surrogate, a value function error term, and an entropy bonus:
        $L _ {t} ^ {C L I P + V F + S} (\theta) = \hat {\mathbb {E}} _ {t} \left[ L _ {t} ^ {C L I P} (\theta) - c _ {1} L _ {t} ^ {V F} (\theta) + c _ {2} S [ \pi_ {\theta} ] (s _ {t}) \right]$
        where $c_1, c_2$ are coefficients, $S$ is an entropy bonus, and $L_{t}^{VF}$ is a squared-error loss $(V_{\theta}(s_t) - V_t^{\mathrm{targ}})^2$.

4.  **Advantage Estimation:**
    *   The advantage estimator $\hat{A}_t$ is typically computed using a learned state-value function $V(s)$.
    *   Generalized Advantage Estimation (GAE) is used, with $\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \dots + (\gamma \lambda)^{T-t+1} \delta_{T-1}$, where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

**Algorithm 1: PPO, Actor-Critic Style**
For each iteration:
1.  For each of $N$ parallel actors, run policy $\pi_{\theta_{old}}$ in the environment for $T$ timesteps.
2.  Compute advantage estimates $\hat{A}_{1},\ldots,\hat{A}_{T}$.
3.  Optimize the surrogate loss $L$ with respect to $\theta$, using $K$ epochs of minibatch SGD (e.g., Adam) with minibatch size $M \leq NT$.
4.  Update $\theta_{old} \leftarrow \theta$.

**Key Formulas in LaTeX:**
*   Policy Gradient Estimator: $\hat{g} = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$
*   Policy Gradient Objective: $L^{PG}(\theta) = \hat{\mathbb{E}}_t \left[ \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$
*   TRPO Surrogate Objective: $\underset{\theta}{\text{maximize}} \quad \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right]$
*   TRPO KL Constraint: $\text{subject to} \quad \hat{\mathbb{E}}_t [\text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)]] \leq \delta$
*   TRPO Penalized Objective: $\underset{\theta}{\text{maximize}} \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)] \right]$
*   Conservative Policy Iteration Objective: $L ^ {C P I} (\theta) = \hat {\mathbb {E}} _ {t} \left[ r _ {t} (\theta) \hat {A} _ {t} \right]$
*   Clipped Surrogate Objective: $L ^ {C L I P} (\theta) = \hat {\mathbb {E}} _ {t} \left[ \min (r _ {t} (\theta) \hat {A} _ {t}, \text{clip} (r _ {t} (\theta), 1 - \epsilon , 1 + \epsilon) \hat {A} _ {t}) \right]$
*   KL-Penalized Objective: $L ^ {K L P E N} (\theta) = \hat {\mathbb {E}} _ {t} \left[ \frac {\pi_ {\theta} (a _ {t} \mid s _ {t})}{\pi_ {\theta_ {\mathrm{old}}} (a _ {t} \mid s _ {t})} \hat {A} _ {t} - \beta \text{KL} [ \pi_ {\theta_ {\mathrm{old}}} (\cdot \mid s _ {t}), \pi_ {\theta} (\cdot \mid s _ {t}) ] \right]$
*   Combined Objective: $L _ {t} ^ {C L I P + V F + S} (\theta) = \hat {\mathbb {E}} _ {t} \left[ L _ {t} ^ {C L I P} (\theta) - c _ {1} L _ {t} ^ {V F} (\theta) + c _ {2} S [ \pi_ {\theta} ] (s _ {t}) \right]$
*   Advantage Estimator (Mnih et al. [2016]): $\hat {A} _ {t} = - V \left(s _ {t}\right) + r _ {t} + \gamma r _ {t + 1} + \dots + \gamma^ {T - t + 1} r _ {T - 1} + \gamma^ {T - t} V \left(s _ {T}\right)$
*   GAE Advantage Estimator: $\hat {A} _ {t} = \delta_ {t} + (\gamma \lambda) \delta_ {t + 1} + \dots + \dots + (\gamma \lambda) ^ {T - t + 1} \delta_ {T - 1}$, where $\delta_ {t} = r _ {t} + \gamma V (s _ {t + 1}) - V (s _ {t})$

**Key Quantitative Results and Numbers:**
*   **Continuous Control Benchmarks (MuJoCo):**
    *   Tested on 7 simulated robotics tasks (HalfCheetah, Hopper, InvertedDoublePendulum, InvertedPendulum, Reacher, Swimmer, Walker2d).
    *   Normalized scores (0 for random policy, 1 for best result) averaged over 21 runs (3 seeds x 7 environments).
    *   **Clipping, $\epsilon = 0.2$** achieved the highest score: **0.82**.
    *   Other clipping values: $\epsilon = 0.1$ (0.76), $\epsilon = 0.3$ (0.70).
    *   Adaptive KL: $d_{\text{targ}} = 0.003$ (0.68), $d_{\text{targ}} = 0.01$ (0.74), $d_{\text{targ}} = 0.03$ (0.71).
    *   Fixed KL: $\beta = 0.3$ (0.62), $\beta = 1$ (0.71), $\beta = 3$ (0.72), $\beta = 10$ (0.69).
    *   "No clipping or penalty" resulted in a negative score of **-0.39**, indicating worse than random performance due to destructive updates.
    *   PPO with clipped objective outperformed TRPO, CEM, vanilla policy gradient, A2C, and A2C with trust region on almost all continuous control environments.
*   **Atari Domain (49 games):**
    *   Comparison against A2C and ACER.
    *   **PPO "won" 30 out of 49 games** based on average episode reward over the entire training period (favors fast learning), compared to ACER (18 wins) and A2C (1 win).
    *   **ACER "won" 28 out of 49 games** based on average episode reward over the last 100 episodes (favors final performance), compared to PPO (19 wins) and A2C (1 win). There was 1 tie.
    *   PPO demonstrated significantly better sample complexity than A2C and similar performance to ACER, despite being much simpler.

**Stated Limitations:**
*   The paper notes that while the KL penalty approach (Equation 5) is theoretically justified, choosing a single fixed value for the penalty coefficient $\beta$ is difficult and often leads to suboptimal performance.
*   The parameters for adapting $\beta$ (1.5 and 2) in the adaptive KL penalty method are chosen heuristically, though the algorithm is not very sensitive to them.
*   The initial value of $\beta$ for the adaptive KL penalty is a hyperparameter but is not considered important in practice due to the algorithm's quick adjustment.
*   The paper mentions that clipping in log space was attempted but found to offer no performance improvement.
*   The experiments on continuous control tasks did not share parameters between the policy and value function ($c_1$ irrelevant) and did not use an entropy bonus.
