---
title: Policy gradient methods for LLMs
kind: primer
reference: ../topics/policy-gradient-methods.md
updated: '2026-07-12'
---

# Policy Gradient Methods for LLM Alignment

Policy gradient methods are the workhorse algorithms that turn human preferences into aligned language models. They provide the gradient estimators that let us optimize non-differentiable reward signals—exactly what we need when the "reward" comes from a learned preference model rather than a hand-coded function. By the end of this primer you will understand the theoretical lineage from REINFORCE to PPO, why advantage estimation (GAE) matters, how Christiano et al. (2017) built the RLHF loop that InstructGPT later instantiated for LLMs, and where the field is moving next (DPO, GRPO, distributional critics).

---

## The Core Mechanism: From Return to Gradient

The fundamental problem: we have a stochastic policy $\pi_\theta(a|s)$ (an LLM generating tokens) and a scalar reward $r$ (from a preference model). We want $\nabla_\theta \mathbb{E}[r]$, but $r$ is not differentiable w.r.t. $\theta$. The **Policy Gradient Theorem** (Sutton et al., 2000) solves this by showing the gradient depends on the action-value function $Q^\pi$ but *not* on the derivative of the state distribution:

$$
\frac{\partial \rho}{\partial \theta} = \sum_s d^\pi(s) \sum_a \frac{\partial \pi(s,a)}{\partial \theta} Q^\pi(s,a)
$$

where $d^\pi(s)$ is the (discounted) state visitation distribution. Crucially, $\nabla_\theta d^\pi$ **does not appear**—the gradient is an expectation over trajectories sampled from the current policy, so we can estimate it by Monte Carlo without differentiating through the environment dynamics.

**Intuition**: If an action $a$ in state $s$ leads to higher-than-average return ($Q^\pi(s,a) > V^\pi(s)$), increase its probability; if lower, decrease it. The log-derivative trick $\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$ turns this into a weight update proportional to the **advantage** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$.

**Why this works**: The baseline $V^\pi(s)$ (or any state-dependent function) subtracts out without bias because $\mathbb{E}_{a\sim\pi}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = \nabla_\theta \sum_a \pi_\theta(a|s) b(s) = \nabla_\theta b(s) = 0$. This variance reduction is the bridge to actor-critic methods.

---

## Actor-Critic and the Compatibility Condition

REINFORCE uses the full return $r$ as a Monte Carlo estimate of $Q^\pi$—unbiased but high variance. **Actor-critic** replaces $Q^\pi$ with a learned critic $f_w(s,a)$. For the critic to yield an *unbiased* policy gradient, Sutton et al. (2000) proved it must satisfy the **compatibility condition**:

$$
\frac{\partial f_w(s,a)}{\partial w} = \frac{\partial \pi(s,a)}{\partial \theta} \frac{1}{\pi(s,a)} = \nabla_\theta \log \pi_\theta(a|s)
$$

This forces $f_w$ to be linear in the policy's "score function" features $\nabla_\theta \log \pi_\theta$, and implies the critic should approximate the **advantage** $A^\pi$, not $Q^\pi$ directly (since adding $V(s)$ leaves the gradient unchanged). Konda & Tsitsiklis (2000) relaxed this by projecting $Q^\pi$ onto the span of $\nabla_\theta \log \pi_\theta$, introducing approximation bias but allowing richer architectures. **Modern practice (PPO, SAC) ignores strict compatibility**, using separate neural networks for policy and value function and relying on empirical performance—a pragmatic divergence from the theory.

---

## Generalized Advantage Estimation (GAE)

In practice we don't know $V^\pi$; we learn $V_\phi(s)$ and estimate advantages from TD residuals:

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

The $k$-step estimator $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V$ trades bias for variance as $k$ grows. **GAE($\gamma,\lambda$)** takes an exponentially weighted average over all $k$:

$$
\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^V
$$

- $\lambda=0$: one-step TD residual (low variance, high bias if $V \neq V^\pi$)
- $\lambda=1$: Monte Carlo return (low bias, high variance)

Empirically, $\gamma \in [0.96, 0.995]$, $\lambda \in [0.92, 0.99]$ with $\lambda < \gamma$ works best on continuous control. The value function is trained by minimizing squared error to Monte Carlo returns with a trust-region constraint on value prediction changes.

```python
# GAE computation check (vectorized, matches Schulman et al. 2015)
import numpy as np

def gae(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE advantages. rewards/values: [T] arrays."""
    T = len(rewards)
    deltas = rewards + gamma * np.append(values[1:], 0.0) - values
    advantages = np.zeros(T)
    gae_acc = 0.0
    for t in reversed(range(T)):
        gae_acc = deltas[t] + gamma * lam * gae_acc
        advantages[t] = gae_acc
    returns = advantages + values
    return advantages, returns

# Sanity checks
r = np.array([1.0, 0.0, 0.0])
v = np.array([0.5, 0.3, 0.1])
adv, ret = gae(r, v, gamma=1.0, lam=1.0)  # lambda=1 -> MC returns
assert np.allclose(adv, [1.0 - 0.5, 0.0 - 0.3, 0.0 - 0.1])  # MC advantage = sum(r) - V(s_t)
adv0, _ = gae(r, v, gamma=1.0, lam=0.0)  # lambda=0 -> TD residual
assert np.allclose(adv0, [1.0 + 0.3 - 0.5, 0.0 + 0.1 - 0.3, 0.0 + 0.0 - 0.1])
print("GAE checks passed.")
```

---

## The RLHF Foundation: Christiano et al. (2017)

Before LLMs, Christiano et al. established the **RLHF paradigm**: learn a reward model from human preferences, then optimize a policy via RL on that learned reward. Three asynchronous processes:

1. **Policy Optimization**: A2C (Atari) or TRPO (MuJoCo) maximizes the learned reward $\hat{r}$.
2. **Preference Elicitation**: Humans compare trajectory segments (1–2 sec clips); choices $\mu \in \{1, 2, \text{equal}, \text{incomparable}\}$.
3. **Reward Fitting**: A Bradley-Terry model predicts preferences from cumulative segment rewards:

$$
\hat{P}[\sigma^1 \succ \sigma^2] = \frac{\exp \sum_t \hat{r}(o_t^1, a_t^1)}{\exp \sum_t \hat{r}(o_t^1, a_t^1) + \exp \sum_t \hat{r}(o_t^2, a_t^2)}
$$

Trained via cross-entropy on the preference dataset $\mathcal{D}$, with **ensembling**, $\ell_2$ regularization, 10% uniform noise modeling, and **active querying** (highest ensemble variance).

**Key result**: <1% of environment interactions needed human feedback. On MuJoCo, 700 queries nearly matched true-reward RL; 1,400 synthetic labels occasionally *exceeded* it (better reward shaping). Complex behaviors (backflips, one-legged cheetah) emerged with ~1 hour of human time.

**Critical caveat**: Offline reward training (static dataset, no online queries) failed—the agent exploited the frozen reward model, producing "bizarre" behaviors (e.g., avoiding losing in Pong without scoring). This foreshadows the **reward hacking** problem in LLM RLHF when the RM is trained offline on static preference data.

---

## PPO for LLM Alignment: The InstructGPT Pipeline

InstructGPT (Ouyang et al., 2022) instantiated the Christiano loop for LLMs:
1. **SFT** on human demonstrations.
2. **Reward Model** trained on ranked completions (Bradley-Terry, same math).
3. **PPO** optimizes the SFT policy against the RM with a KL penalty from the SFT model.

The PPO clipped surrogate objective with GAE advantages:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right], \quad r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

For LLMs: state = prompt + generated tokens so far; action = next token; episode = full response. The combined objective adds value loss and entropy bonus:

$$
L_t^{\text{CLIP+VF+S}}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{\text{CLIP}}(\theta) - c_1 L_t^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t) \right] - \beta\,\text{KL}[\pi^{\text{SFT}}(\cdot|s_t), \pi_\theta(\cdot|s_t)]
$$

**Results**: 1.3B InstructGPT preferred over 175B GPT-3; 175B InstructGPT preferred over 175B GPT-3 $85\pm3\%$; TruthfulQA truthfulness doubled, hallucination dropped 41% → 21%.

**Load-bearing disagreement**: The original PPO paper proposed *either* clipping *or* an adaptive KL penalty ($\beta$ tuned to target KL $d_{\text{targ}}$). InstructGPT and most open-source implementations (TRL, OpenRLHF) use **clipping + a fixed KL penalty** simultaneously. The theoretical interaction is uncharacterized; the combination is empirically tuned.

---

## Load-Bearing Caveats

1. **Critic compatibility vs. practice**: Theory (Sutton et al.) demands the critic approximate the advantage with a specific linear architecture for unbiased gradients. Modern deep actor-critic (PPO, SAC) uses separate MLPs for policy and value, violating compatibility. This works empirically but loses the unbiased-gradient guarantee—bias is controlled by clipping and KL penalties instead.

2. **Offline reward models invite hacking**: Christiano et al. showed that a static RM (no online queries) leads to exploitation. Modern LLM RLHF almost always trains the RM offline on a fixed preference dataset. The KL penalty and clipping mitigate but do not eliminate this; "reward model over-optimization" remains an active research problem.

---

## Current Status

PPO+GAE remains the default RLHF algorithm (Llama 2, Zephyr, TRL), but the consensus is shifting toward critic-free (GRPO) and RL-free (DPO, IPO, KTO, SimPO) alternatives that simplify or eliminate the advantage estimation pipeline, especially for discrete, massive action spaces where reward models are noisy.

**Full reference**: See the companion reference article for exhaustive citations, variant catalogs, hyperparameter tables, and the distributional GAE (DGAE) extension.

---
*Full reference (citations, derivations, variants):* [Policy gradient methods for LLMs](../topics/policy-gradient-methods.md)
