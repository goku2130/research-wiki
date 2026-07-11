---
title: Policy gradient methods for LLMs
maturity: developing
updated: '2026-07-11'
sources:
- link:simple-statistical-gradient-following-al
- papers:policy-gradient-methods-for-reinforcemen
- proceedings:actor-critic-algorithms
- arxiv:1506.02438
- arxiv:1707.06347
- arxiv:1801.01290
- arxiv:2203.02155
- arxiv:2507.17530
open_questions:
- How does the discrete, massive action space of LLMs (vocabulary size ~100k) affect
  the bias-variance tradeoff in GAE compared to continuous control, and does the standard
  $\lambda \approx 0.95$ remain optimal?
- The compatibility condition requires the critic to approximate the advantage function;
  in PPO for LLMs, the value head predicts $V(s)$ not $A(s,a)$—does this mismatch
  introduce systematic gradient bias, and how large is it in practice?
- Distributional critics (DGAE) capture return distribution shape; for LLMs with sparse,
  high-variance rewards (e.g., 0/1 correctness on math), could distributional advantages
  provide more stable learning signals than scalar GAE?
- The KL penalty in RLHF is typically applied per-token; does this induce a different
  implicit constraint than the per-episode KL constraint analyzed in the policy gradient
  convergence proofs (Konda & Tsitsiklis, 2000)?
---

Policy gradient methods form the theoretical backbone of reinforcement learning from human feedback (RLHF) for large language models, providing the gradient estimators that enable direct optimization of non-differentiable reward signals. This article traces the lineage from the foundational REINFORCE estimator through actor-critic architectures, advantage estimation techniques including GAE, and their modern instantiations in LLM alignment pipelines.

## Foundations: REINFORCE and the Policy Gradient Theorem

The REINFORCE algorithm, introduced by Williams (1992), established the fundamental identity that the gradient of expected reward with respect to policy parameters $\theta$ can be expressed as an expectation over trajectories sampled from the policy [source:link:simple-statistical-gradient-following-al]:

$$
\nabla_\theta \mathbb{E}[r] = \mathbb{E}\left[ r \nabla_\theta \log \pi_\theta(\tau) \right]
$$

where $\tau = (s_0, a_0, s_1, a_1, \dots)$ denotes a trajectory and $r$ is the total return. For a stochastic policy $\pi_\theta(a|s)$, this yields the per-step update:

$$
\Delta \theta = \alpha (r - b) \nabla_\theta \log \pi_\theta(a|s)
$$

where $b$ is a baseline that reduces variance without introducing bias, provided $b$ does not depend on the action $a$ [source:link:simple-statistical-gradient-following-al]. For Bernoulli stochastic units with output probability $p = f(\text{net})$, the eligibility term becomes $(y - p) \frac{\partial \text{net}}{\partial w}$; for Gaussian units with mean $\mu = f(\text{net})$ and fixed variance $\sigma^2$, it becomes $\frac{(y - \mu)}{\sigma^2} \frac{\partial \mu}{\partial w}$ [source:link:simple-statistical-gradient-following-al].

Sutton et al. (2000) generalized this to the **Policy Gradient Theorem** for Markov decision processes, showing that for either the average-reward objective $\rho(\pi) = \lim_{n\to\infty} \frac{1}{n}\mathbb{E}[\sum_{k=1}^n r_k]$ or the discounted start-state objective $\rho(\pi) = \mathbb{E}[\sum_{k=0}^\infty \gamma^k r_{k+1} | s_0]$, the gradient takes the form [source:papers:policy-gradient-methods-for-reinforcemen]:

$$
\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta} Q^\pi(s,a)
$$

where $d^\pi(s)$ is the stationary (or discounted) state distribution under $\pi$. Crucially, this gradient does **not** depend on the derivative of the state distribution with respect to $\theta$, enabling unbiased estimation via sampling [source:papers:policy-gradient-methods-for-reinforcemen].

## Actor-Critic Architecture and Compatibility

The REINFORCE estimator suffers from high variance because it uses the full return $r$ as a Monte Carlo estimate of $Q^\pi$. Actor-critic methods replace this with a learned critic $f_w(s,a) \approx Q^\pi(s,a)$, yielding the update:

$$
\Delta \theta = \alpha \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta} f_w(s,a)
$$

Konda and Tsitsiklis (2000) analyzed two-time-scale actor-critic algorithms where the critic (parameters $w$) updates on a faster timescale than the actor (parameters $\theta$) [source:proceedings:actor-critic-algorithms]. They showed that for the critic to provide an unbiased gradient estimate, it must satisfy a **compatibility condition**:

$$
\frac{\partial f_w(s,a)}{\partial w} = \frac{\partial \pi(s,a)}{\partial \theta} \frac{1}{\pi(s,a)}
$$

This implies $f_w$ should approximate the **advantage function** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ rather than $Q^\pi$ directly, since adding any state-dependent baseline $V(s)$ leaves the gradient unchanged [source:papers:policy-gradient-methods-for-reinforcemen]. The critic is trained to minimize a weighted mean-squared error where the weighting ensures the error is orthogonal to the policy gradient features:

$$
\sum_s d^\pi(s) \sum_a \pi(s,a) [Q^\pi(s,a) - f_w(s,a)] \frac{\partial f_w(s,a)}{\partial w} = 0
$$

[source:papers:policy-gradient-methods-for-reinforcemen]. Konda and Tsitsiklis proved convergence to a local optimum under the two-timescale stepsize conditions $\sum \beta_k = \infty, \sum \beta_k^2 < \infty$ for the actor and $\sum \gamma_k = \infty, \sum \gamma_k^2 < \infty$ for the critic, with $\beta_k / \gamma_k \to 0$ [source:proceedings:actor-critic-algorithms].

**Disagreement on critic parameterization**: Sutton et al. (2000) require the compatibility condition $\frac{\partial f_w}{\partial w} = \frac{\partial \pi}{\partial \theta} \frac{1}{\pi}$, which forces $f_w$ to be linear in the policy's score function features [source:papers:policy-gradient-methods-for-reinforcemen]. Konda and Tsitsiklis (2000) instead project the true $q^\pi$ onto the span of $\psi^\theta = \nabla_\theta \log \pi_\theta$, using a linear architecture $Q_r^\theta(x,u) = \sum_j r_j \phi_j^\theta(x,u)$ where $\text{span}\{\phi_j\}$ contains $\text{span}\{\psi_j\}$ [source:proceedings:actor-critic-algorithms]. The latter is more flexible but introduces approximation bias if the projection is imperfect. Modern deep actor-critic methods (e.g., PPO, SAC) typically use separate neural networks for policy and value function without enforcing strict compatibility, relying on empirical performance rather than theoretical guarantees [source:arxiv:1707.06347][source:arxiv:1801.01290].

## Advantage Estimation and GAE

In practice, the advantage function $A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$ is estimated using a learned value function $V_\phi(s)$. Schulman et al. (2015) introduced **Generalized Advantage Estimation (GAE)**, which interpolates between high-bias/low-variance and low-bias/high-variance estimators via a parameter $\lambda \in [0,1]$ [source:arxiv:1506.02438]. Define the TD residual:

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

The $k$-step advantage estimator is $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$. GAE($\gamma,\lambda$) takes an exponentially weighted average:

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^V
$$

Special cases: $\lambda=0$ gives the one-step TD residual $\delta_t$ (low variance, high bias if $V \neq V^\pi$); $\lambda=1$ gives the Monte Carlo return $\sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t)$ (low bias, high variance) [source:arxiv:1506.02438]. The paper also introduces a discount parameter $\gamma \leq 1$ distinct from the MDP discount, which downweights delayed effects to further reduce variance at the cost of bias [source:arxiv:1506.02438].

Empirically, on MuJoCo continuous control tasks, the best $\gamma$ values were in $[0.96, 0.995]$ and best $\lambda$ in $[0.92, 0.99]$, with $\lambda$ typically lower than $\gamma$ [source:arxiv:1506.02438]. The value function was trained by minimizing squared error to Monte Carlo returns with a trust-region constraint on the KL divergence between old and new value predictions [source:arxiv:1506.02438].

**Distributional extension**: A 2025 paper extends GAE to distributional RL by defining a Wasserstein-like directional metric $d(F_U, G_V) = \inf_{U,V}(U-V) = \int_0^1 L(F_U^{-1}(q) - G_V^{-1}(q)) dq$ that captures both distance and "superiority" between return distributions [source:arxiv:2507.17530]. The distributional TD error becomes $\delta^G(s_t,a_t) = r(s_t,a_t) + d(\gamma G(S_{t+1}), G(s_t))$, and DGAE is defined analogously. On MuJoCo tasks, distributional PPO (DPPO) and TRPO (DTRPO) outperformed their scalar baselines, though A2C/DA2C remained weak [source:arxiv:2507.17530]. The directional metric ignores shape similarity (e.g., variance differences), but this occurred in only ~0.093% of cases on Hopper [source:arxiv:2507.17530].

## Application to LLM Fine-tuning: PPO and RLHF

The InstructGPT paper (Ouyang et al., 2022) instantiated policy gradient methods for LLM alignment via a three-stage pipeline: (1) supervised fine-tuning (SFT) on human demonstrations, (2) reward model (RM) training on ranked comparisons, (3) PPO optimization of the SFT policy against the RM with a KL penalty from the SFT model [source:arxiv:2203.02155].

The PPO objective used is the clipped surrogate [source:arxiv:1707.06347]:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ and $\hat{A}_t$ is computed via GAE [source:arxiv:1707.06347]. For LLMs, the "state" $s_t$ is the prompt plus generated tokens so far, the "action" $a_t$ is the next token, and the episode terminates at the end of the response. The KL penalty term $-\beta \text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]$ regularizes against reward hacking [source:arxiv:2203.02155]. The combined objective includes a value function loss and entropy bonus:

$$
L_t^{\text{CLIP+VF+S}}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{\text{CLIP}}(\theta) - c_1 L_t^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

[source:arxiv:1707.06347]. InstructGPT used $\epsilon=0.2$, GAE with $\gamma=1, \lambda=0.95$, and a per-token KL coefficient $\beta$ [source:arxiv:2203.02155]. The PPO-ptx variant added a pretraining gradient term $\gamma \mathbb{E}_{x\sim D_{\text{pretrain}}}[\log \pi_\theta(x)]$ to mitigate the alignment tax [source:arxiv:2203.02155].

**Quantitative results**: On continuous control, PPO with $\epsilon=0.2$ achieved a normalized score of 0.82 vs. TRPO's 0.77; on Atari, PPO won 30/49 games by average reward during training vs. ACER's 18 [source:arxiv:1707.06347]. For InstructGPT, the 1.3B model was preferred over 175B GPT-3, and 175B InstructGPT was preferred over 175B GPT-3 $85\pm3\%$ of the time [source:arxiv:2203.02155]. TruthfulQA truthfulness doubled; hallucination rate dropped from 41% to 21% on closed-domain tasks [source:arxiv:2203.02155].

**Disagreement on KL penalty form**: The original PPO paper proposed both clipping and an adaptive KL penalty $L^{\text{KLPEN}} = \hat{\mathbb{E}}_t[r_t(\theta)\hat{A}_t - \beta \text{KL}]$ with $\beta$ adjusted to target a KL divergence $d_{\text{targ}}$ [source:arxiv:1707.06347]. InstructGPT uses a fixed per-token KL coefficient $\beta$ added to the reward, not the clipped objective [source:arxiv:2203.02155]. Later work (e.g., PPO implementations in TRL, OpenRLHF) often uses the clipped objective **with** a fixed KL penalty term, combining both mechanisms. The theoretical interaction between clipping and KL penalties is not fully characterized; Schulman et al. note clipping in log space offered no improvement [source:arxiv:1707.06347], but the combined effect in LLM settings is empirically tuned.

## Current Status and Trajectory

Policy gradient methods—specifically PPO with GAE—remain the **default** algorithm for the RL step of RLHF in open-source and industrial LLM alignment (e.g., Llama 2, Zephyr, TRL library). However, the trajectory shows **rising interest in alternatives** that avoid the complexity and instability of on-policy PPO:

- **Direct Preference Optimization (DPO)** and variants (IPO, KTO, SimPO) replace the RL loop with a direct supervised loss on preference pairs, eliminating the need for a separate RM, value function, and GAE estimation [source:arxiv:2203.02155] (see sibling article on DPO).
- **Group Relative Policy Optimization (GRPO)** removes the critic entirely, estimating advantages via group statistics over multiple responses per prompt, reducing memory and compute [source:arxiv:2507.17530] (see sibling article on GRPO).
- **Off-policy methods (SAC, TD3)** have seen limited adoption for LLM fine-tuning due to the difficulty of replay buffers with long sequences and the non-stationarity of the reward model, though they remain standard in continuous control [source:arxiv:1801.01290].
- **Distributional critics (DGAE)** are a recent research direction (2025) with promising MuJoCo results but **not widely reported** in LLM settings; the quantile-Huber loss and inverse CDF networks add significant complexity for unclear gains in discrete token spaces [source:arxiv:2507.17530].

The field is **not abandoning** policy gradients—PPO remains the benchmark—but the **consensus is shifting** toward methods that simplify or eliminate the critic and advantage estimation pipeline, especially for instruction-following where reward models are noisy and the action space is discrete and massive.

## Key Takeaways

- REINFORCE provides the unbiased but high-variance foundation: $\nabla_\theta J = \mathbb{E}[R \nabla_\theta \log \pi_\theta]$ [source:link:simple-statistical-gradient-following-al].
- The Policy Gradient Theorem (Sutton et al., 2000) shows the gradient depends on $Q^\pi$ but not $\nabla_\theta d^\pi$, enabling sampling-based estimation [source:papers:policy-gradient-methods-for-reinforcemen].
- Actor-critic methods reduce variance by learning a critic; compatibility requires the critic to approximate the advantage function, not $Q^\pi$ [source:papers:policy-gradient-methods-for-reinforcemen][source:proceedings:actor-critic-algorithms].
- GAE($\gamma,\lambda$) interpolates between TD($\lambda=0$) and MC($\lambda=1$) advantage estimates via exponentially weighted TD residuals; $\lambda \approx 0.95-0.99$ is standard [source:arxiv:1506.02438].
- PPO's clipped surrogate objective with GAE advantages and KL regularization is the dominant RLHF algorithm; InstructGPT demonstrated 1.3B aligned models outperforming 175B unaligned ones [source:arxiv:1707.06347][source:arxiv:2203.02155].
- Distributional GAE (DGAE) extends advantage estimation to return distributions using a directional Wasserstein metric; improves continuous control but untested at LLM scale [source:arxiv:2507.17530].
- The ecosystem is moving toward critic-free (GRPO) or RL-free (DPO) alternatives, but PPO+GAE remains the reference implementation and theoretical touchstone.

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [KL regularization in RLHF](kl-regularization.md)
- [MDP formulation of LLM generation](mdp-formulation.md)
- [RL for LLMs — overview](rl-for-llms-overview.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)

## References
- [source:link:simple-statistical-gradient-following-al] [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/article/10.1007/BF00992696)
- [source:papers:policy-gradient-methods-for-reinforcemen] [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.neurips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [source:proceedings:actor-critic-algorithms] [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1786-actor-critic-algorithms.pdf)
- [source:arxiv:1506.02438] [High-Dimensional Continuous Control Using Generalized Advantage Estimation and Trust Region Policy Optimization](https://arxiv.org/pdf/1506.02438)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)
- [source:arxiv:1801.01290] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)
- [source:arxiv:2507.17530] [Generalized Advantage Estimation for Distributional Policy Gradients](https://arxiv.org/abs/2507.17530v1)
