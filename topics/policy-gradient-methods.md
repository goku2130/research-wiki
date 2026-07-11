---
title: Policy gradient methods for LLMs
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:simple-statistical-gradient-following-al
- arxiv:1506.02438
- arxiv:1707.06347
- arxiv:2203.02155
- arxiv:2305.18290
- arxiv:2402.03300
- arxiv:1502.05477
- arxiv:policy-gradient-methods-for-reinforcemen
open_questions:
- How does the generalization of DPO-trained policies compare to PPO-trained policies
  on truly out-of-distribution prompts? [source:arxiv:2305.18290]
- Can the group-relative advantage estimation in GRPO maintain stability across diverse
  non-reasoning tasks (e.g., creative writing) where reward variance is higher? [source:arxiv:2402.03300]
- Is there a principled way to automate the tuning of $\gamma$ and $\lambda$ in GAE
  to avoid manual empirical search? [source:arxiv:1506.02438]
---

Policy gradient methods optimize the expected reward of a language model by adjusting its parameters in the direction of the gradient of the performance objective. In the context of Large Language Models (LLMs), these methods facilitate the alignment of model outputs with human preferences through Reinforcement Learning from Human Feedback (RLHF) [source:arxiv:2203.02155].

## Core Mechanics of Policy Gradients

The fundamental goal of policy gradient methods is to maximize the expected cumulative reward $\eta(\pi) = \mathbb{E}_{\pi} [\sum_{t=0}^{\infty} \gamma^t r_t]$. The gradient of this objective takes the general form:

$$
g = \mathbb{E} \left[ \sum_{t=0}^{\infty} \Psi_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

where $\pi_\theta$ is the policy (the LLM), and $\Psi_t$ is a scalar representing the "goodness" of the action $a_t$ taken at time $t$ [source:arxiv:1506.02438].

### The Bias-Variance Tradeoff
A central challenge in policy gradients is the choice of $\Psi_t$. Using the actual cumulative return (as in vanilla REINFORCE) provides an unbiased estimate of the gradient but suffers from high variance that scales with the time horizon [source:arxiv:1506.02438]. Conversely, using a learned value function $V(s)$ to estimate the return reduces variance but introduces bias, which can prevent the policy from converging to optimal policies [source:arxiv:1506.02438].

## Advantage Estimation and GAE

To mitigate the bias-variance tradeoff, researchers use the **Advantage Function** $A(s, a) = Q(s, a) - V(s)$, which measures how much better an action is compared to the average action at that state [source:arxiv:1506.02438].

### Generalized Advantage Estimation (GAE)
GAE provides a parameterized way to interpolate between low-bias/high-variance and high-bias/low-variance estimates [source:arxiv:1506.02438]. The GAE estimator is defined as an exponentially weighted sum of Temporal Difference (TD) residuals:

$$
\hat{A}_{t}^{\mathrm{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}^{V}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual [source:arxiv:1506.02438][source:arxiv:1707.06347].

The hyperparameters control the tradeoff:
*   **$\lambda = 0$**: Becomes a 1-step TD estimate (low variance, high bias) [source:arxiv:1506.02438].
*   **$\lambda = 1$**: Becomes an empirical return minus a baseline (high variance, low bias) [source:arxiv:1506.02438].
*   **Optimal Range**: Empirical results in continuous control suggest $\gamma \in [0.96, 0.995]$ and $\lambda \in [0.92, 0.99]$ [source:arxiv:1506.02438].

## Actor-Critic Architectures

LLM alignment typically employs an actor-critic framework where two separate (or shared) networks are trained:
1.  **The Actor ($\pi_\theta$):** The LLM being optimized to generate tokens that maximize reward [source:arxiv:1707.06347].
2.  **The Critic ($V_\phi$):** A value function that predicts the expected return from a given state to reduce gradient variance [source:arxiv:1506.02438].

### Trust Region Policy Optimization (TRPO)
To prevent unstable updates and ensure monotonic improvement, TRPO enforces a hard constraint on the average KL divergence between the old and new policy:

$$
\underset{\theta}{\text{maximize }} L_{\theta_{\text{old}}}(\theta) \quad \text{subject to } \overline{D}_{\mathrm{KL}}^{\rho_{\theta_{\text{old}}}}(\theta_{\text{old}}, \theta) \leq \delta
$$

TRPO uses a second-order optimization method involving the Fisher Information Matrix and conjugate gradient descent to solve this constrained problem [source:arxiv:1502.05477].

### Proximal Policy Optimization (PPO)
PPO simplifies TRPO by using a first-order optimization framework. It introduces a **clipped surrogate objective** to discourage the policy from moving too far from $\pi_{\theta_{\text{old}}}$:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio [source:arxiv:1707.06347].

| Feature | TRPO | PPO (Clipped) |
| :--- | :--- | :--- |
| **Optimization** | Second-order (Conjugate Gradient) | First-order (Adam/SGD) |
| **Constraint** | Hard KL constraint $\delta$ | Clipped probability ratio $\epsilon$ |
| **Complexity** | High (Hessian-vector products) | Low (Standard backprop) |
| **Stability** | Monotonic improvement guarantee | Empirical stability via clipping |
| **Compatibility** | Incompatible with dropout/sharing | Compatible with most architectures |
| **Source** | [source:arxiv:1502.05477] | [source:arxiv:1707.06347] |

## Application to LLM Alignment (RLHF)

In the RLHF pipeline, policy gradient methods are used to align an SFT (Supervised Fine-Tuning) model with a Reward Model (RM) [source:arxiv:2203.02155].

### The RLHF Objective
The objective is to maximize the reward $r_\theta(x, y)$ while penalizing the model for drifting too far from the SFT reference policy $\pi^{\text{SFT}}$ to prevent "reward hacking" [source:arxiv:2203.02155]:

$$
\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\frac{\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x)}{\pi^ {\mathrm{SFT}} (y \mid x)}\right) \right]
$$

To combat the "alignment tax" (performance degradation on general tasks), the **PPO-ptx** variant mixes in pretraining distribution gradients:

$$
\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\frac{\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x)}{\pi^ {\mathrm{SFT}} (y \mid x)}\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]
$$

where $\gamma$ controls the strength of the pretraining gradient [source:arxiv:2203.02155].

## Modern Evolutions: DPO and GRPO

Recent methods seek to remove the instability and overhead of the actor-critic loop.

### Direct Preference Optimization (DPO)
DPO eliminates the explicit reward model and the PPO loop entirely. It uses a mathematical reparameterization to express the reward function in terms of the optimal policy $\pi_r$ and a reference policy $\pi_{\text{ref}}$ [source:arxiv:2305.18290]:

$$
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)
$$

By substituting this into a Bradley-Terry preference model, DPO optimizes the policy using a simple binary cross-entropy loss on preferred ($y_w$) and dispreferred ($y_l$) pairs [source:arxiv:2305.18290].

### Group Relative Policy Optimization (GRPO)
GRPO reduces overhead by removing the critic network entirely. Instead of using a value function $V(s)$ to estimate the baseline for advantages, GRPO samples a group of $G$ outputs per prompt and normalizes the rewards within that group to estimate the advantage [source:arxiv:2402.03300].

## Disagreements and Divergent Findings

*   **Clipped vs. KL-Penalty in PPO**: While PPO supports both a clipped objective and an adaptive KL-penalized objective ($L^{KLPEN}$), the authors of PPO report that the KL-penalty variant consistently underperformed the clipped objective across all tested configurations [source:arxiv:1707.06347].
*   **DPO vs. PPO Performance**: DPO claims that its reward-KL frontier strictly dominates PPO in sentiment control and outperforms PPO in summarization win rates [source:arxiv:2305.18290]. However, the authors admit that the generalization of DPO policies out-of-distribution compared to explicit reward-based methods is not fully characterized [source:arxiv:2305.18290].

## Current status and trajectory

*   **PPO**: Remains a **default** for high-budget RLHF pipelines [source:arxiv:2203.02155], but is increasingly viewed as computationally cumbersome due to the need for multiple models (policy, value, reference, reward) in memory [source:arxiv:2305.18290].
*   **DPO**: **Rising** rapidly as a stable, memory-efficient alternative that replaces RL with a classification task [source:arxiv:2305.18290].
*   **GRPO**: **Emerging**, specifically for reasoning-heavy models (e.g., DeepSeekMath), where removing the critic reduces the memory footprint for large-scale RL [source:arxiv:2402.03300].
*   **Trajectory**: The field is moving away from complex actor-critic setups toward preference-based optimization (DPO) or group-relative baselines (GRPO) to improve stability and reduce compute [source:arxiv:2305.18290][source:arxiv:2402.03300].

## Key takeaways

*   **Variance Reduction**: GAE is the standard for balancing bias and variance in policy gradients via the $\lambda$ parameter [source:arxiv:1506.02438].
*   **Stability**: PPO's clipped objective provides a first-order approximation of TRPO's trust region, preventing catastrophic policy collapse [source:arxiv:1707.06347].
*   **KL Regularization**: In LLMs, KL penalties are essential to prevent the model from exploiting the reward model (reward hacking) and to preserve the capabilities of the SFT model [source:arxiv:2203.02155].
*   **Efficiency Shift**: DPO and GRPO represent a shift toward removing the "Critic" network to save memory and simplify the optimization objective [source:arxiv:2305.18290][source:arxiv:2402.03300].

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [KL regularization in RLHF](kl-regularization.md)
- [RL for LLMs — overview](rl-for-llms-overview.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [The alignment tax](alignment-tax.md)

## References
- [source:arxiv:simple-statistical-gradient-following-al] [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://arxiv.org/abs/cond-mat/9212003)
- [source:arxiv:1506.02438] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2402.03300] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [source:arxiv:1502.05477] [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [source:arxiv:policy-gradient-methods-for-reinforcemen] [Policy gradient methods for reinforcement learning with function approximation](https://arxiv.org/abs/cs/0602081)
