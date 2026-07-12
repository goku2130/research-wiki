---
title: GRPO (Group Relative Policy Optimization)
kind: primer
reference: ../topics/grpo.md
updated: '2026-07-12'
---

# Group Relative Policy Optimization (GRPO)

**Scaffold.** GRPO is a critic-free policy gradient algorithm that replaces PPO's value network with a group-relative baseline computed from multiple sampled responses per prompt. By the end of this primer you will understand: (1) how the group statistics produce a per-completion advantage without any learned critic, (2) the exact objective function and why its KL estimator is numerically stable, (3) why GRPO pairs naturally with verifiable rewards (RLVR) but struggles on long-horizon agentic tasks, and (4) the key empirical caveats — especially the classical-control evidence that critic-free methods fail without early termination. This connects to PPO (the actor-critic baseline it replaces), RLVR (the reward paradigm that makes it viable), and KL regularization (the trust-region mechanism it internalizes).

---

## Core mechanism: the group as its own baseline

In standard PPO, a critic network $V_\phi(s_t)$ estimates the expected return from each token position, providing a per-step baseline for advantage estimation. GRPO observes that for **episodic, prompt-conditioned generation** — where every trajectory ends at an EOS token and a scalar reward is available — we can compute a baseline directly from the *empirical distribution of rewards* across a group of completions for the same prompt.

For a prompt $q$, sample $G$ completions $\{o_i\}_{i=1}^G$ from the current policy $\pi_{\theta_{\text{old}}}$. Each receives a scalar reward $r_i$ (from a verifier or reward model). The group mean $\mu_r$ and standard deviation $\sigma_r$ define a **normalized advantage** shared by every token in completion $i$:

$$
\widetilde{r}_i = \frac{r_i - \mu_r}{\sigma_r}, \qquad \hat{A}_{i,t} = \widetilde{r}_i \quad \text{(outcome supervision)}
$$

This is the entire critic replacement: no value network, no GAE, no per-token estimation. The intuition is simple — if a completion scores *above the group average*, all its tokens get positive credit; if *below*, all get negative credit. The normalization by $\sigma_r$ automatically scales the learning signal to the group's spread, preventing reward-scale drift.

**Why this works for LLMs.** LLM generation has a natural horizon (the EOS token). The terminal reward cleanly separates "good" from "bad" completions *within the same prompt context*, so the group statistics are a low-variance, prompt-conditioned baseline. In continuing control tasks without early termination (e.g., HalfCheetah), this signal dilutes — a finding we return to under caveats.

**Process supervision variant.** When a process reward model provides step-level rewards $r_i^{\text{index}(j)}$, GRPO normalizes them across the group and defines the token-level advantage as the sum of normalized *future* step rewards:

$$
\widetilde{r}_i^{\text{index}(j)} = \frac{r_i^{\text{index}(j)} - \mu_R}{\sigma_R}, \qquad \hat{A}_{i,t} = \sum_{\text{index}(j) \ge t} \widetilde{r}_i^{\text{index}(j)}
$$

DeepSeekMath reports outcome supervision is preferred in practice — the marginal gains don't justify the overhead — but process supervision can accelerate learning on complex multi-step reasoning.

---

## Objective function: PPO clipping + direct KL penalty

GRPO adopts PPO's clipped surrogate objective with the group-relative advantage, and adds a **direct KL penalty against a reference policy** $\pi_{\text{ref}}$ (typically the SFT model, updated each iteration in DeepSeek's recipe):

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left( \rho_{i,t} \hat{A}_{i,t},\; \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \right) - \beta \, \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right\} \right]
$$

where $\rho_{i,t} = \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}$.

The KL term uses an **unbiased, non-negative estimator** that avoids log-probability underflow:

$$
\mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - 1
$$

This estimator is derived from the identity $D_{\text{KL}}(p\|q) = \mathbb{E}_q[q/p - \log(q/p) - 1]$ and is numerically stable even when $\pi_\theta$ assigns near-zero probability to tokens that $\pi_{\text{ref}}$ assigns mass to — a common failure mode of the naive $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ form.

**Moving reference policy.** Unlike standard PPO where $\pi_{\text{ref}}$ is frozen as the initial SFT model, DeepSeek updates $\pi_{\text{ref}} \leftarrow \pi_\theta$ each iteration. This implements a trust region around the *most recent* policy rather than the original SFT model. The long-term effects on mode collapse or catastrophic forgetting are not analyzed in the literature.

---

## Runnable check: group-relative advantage computation

```python
import numpy as np

def grpo_advantages(rewards: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute GRPO group-relative advantages (outcome supervision).
    
    Args:
        rewards: Shape (G,) — scalar reward per completion in the group.
        eps: Small constant to avoid division by zero.
    Returns:
        advantages: Shape (G,) — normalized advantage per completion.
    """
    assert rewards.ndim == 1, "rewards must be 1D (group dimension)"
    G = rewards.shape[0]
    assert G >= 2, "need at least 2 completions for std"
    
    mu = rewards.mean()
    sigma = rewards.std(ddof=0) + eps  # population std per paper
    adv = (rewards - mu) / sigma
    
    # Sanity checks
    assert np.allclose(adv.mean(), 0.0, atol=1e-6), "advantages must center at 0"
    assert np.allclose(adv.std(ddof=0), 1.0, atol=1e-6), "advantages must have unit variance"
    return adv

# Quick test
r = np.array([0.0, 0.5, 1.0, 1.0, 0.5])  # 5 completions
a = grpo_advantages(r)
print("Rewards:", r)
print("Advantages:", a.round(4))
print("Mean:", a.mean().round(6), "Std:", a.std(ddof=0).round(6))
```

**What this verifies:** The advantage computation centers at zero with unit variance by construction — the group *is* the baseline. Every token in completion $i$ receives the same scalar $\hat{A}_i$.

---

## Load-bearing disagreements & caveats

### 1. Group size: theory vs. practice vs. classical control
The DeepSeekMath paper uses $G=64$; the YouTube walkthrough cites "typically 4–8"; Snorkel reports "8–64." The classical control study (arXiv:2511.03527) found **smaller groups ($G=8$) outperformed larger ones ($G=128$)**, even controlling for update frequency — possibly because their grouping strategy mixed unrelated episodes from parallel environments, unlike prompt-conditioned grouping in LLMs. The variance/compute trade-off is not theoretically characterized; treat $G$ as a hyperparameter tuned to your reward variance and compute budget.

### 2. Critic-free works for LLMs *because* of early termination
The systematic classical-control study (CartPole, Acrobot, MountainCarContinuous, HalfCheetah, Humanoid) found **PPO with a learned critic substantially outperforms all critic-free baselines on long-horizon tasks without early termination** (HalfCheetah, Humanoid). The only exception was CartPole, where critic-free methods often exceeded PPO (attributed to PPO overfitting/collapse in this short-horizon environment). The authors identify **early termination** as the critical factor: environments with natural termination (CartPole, Acrobot) allow critic-free methods to extract learning signals from the return distribution, while continuing tasks dilute the signal. LLM generation has a natural horizon (EOS token), which likely explains why GRPO succeeds there despite failing in continuous control. **Do not assume GRPO is universally superior to actor-critic methods.**

---

## Current status
GRPO is the **dominant optimizer for verifiable-reward reasoning (RLVR)** — core to DeepSeek-R1, Qwen-3, Olmo-3, and reportedly OpenAI o1/o3/o4 and Gemini 3 — but **not a settled universal replacement for PPO** in preference-based RLHF or long-horizon agentic tasks.

**Full reference:** See the linked reference article for exhaustive variant catalogs, hyperparameter tables, empirical results (DeepSeekMath 7B: MATH 46.8% → 51.7%, GSM8K 82.9% → 88.2%), classical control benchmarks, and the complete bibliography (DeepSeekMath arXiv:2402.03300, classical control study arXiv:2511.03527, and secondary explainers).

---
*Full reference (citations, derivations, variants):* [GRPO (Group Relative Policy Optimization)](../topics/grpo.md)
