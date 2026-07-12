---
title: PPO for LLM fine-tuning (RLHF)
kind: primer
reference: ../topics/ppo-for-llms.md
updated: '2026-07-12'
---

# PPO for LLM Fine-Tuning (RLHF)

**What this is:** Proximal Policy Optimization (PPO) is the canonical online reinforcement learning algorithm for aligning large language models via Reinforcement Learning from Human Feedback (RLHF). It adapts the clipped surrogate objective from continuous control to a contextual bandit where each token is an action, the reward is a single scalar from a learned reward model at sequence end, and two distinct regularizers — clipping and a KL penalty — stabilize training. By the end you will understand how the PPO objective decomposes into a per-token policy ratio, a generalized advantage estimate, and a KL anchor to the supervised model; why reward-model overoptimization is a structural problem; and where the method stands relative to offline alternatives like DPO and P3O.

**What it connects to:** The policy-gradient theorem, Bradley-Terry reward modeling, KL-regularized optimization, and the broader shift from online RLHF to offline preference optimization.

---

## Core Mechanism: Two Regularizers, One Bandit

PPO for LLMs solves a **contextual bandit**, not a multi-step MDP. The prompt $x$ is the context; the response $y = (y_1,\dots,y_T)$ is the trajectory; each token $y_t$ is an action drawn from $\pi_\theta(y_t|x,y_{<t})$. The reward $r_\psi(x,y)$ arrives only at the end. The policy gradient for a full sequence is

$$
\nabla_\theta J(\theta) = \mathbb{E}_{x,y\sim\pi_\theta}\Bigl[ \sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t|x,y_{<t}) \cdot \hat{A}_t \Bigr],
$$

where $\hat{A}_t$ is the advantage. Because the reward is sparse and terminal, Generalized Advantage Estimation (GAE) collapses to a single-step form: $\hat{A}_t \approx r_\psi(x,y) - V_\phi(x)$ for all $t$, with $V_\phi$ a critic initialized from the reward model backbone.

**The clipped surrogate objective** replaces the hard KL constraint of TRPO with a pessimistic bound on the policy ratio $r_t(\theta) = \pi_\theta(y_t|x,y_{<t}) / \pi_{\theta_{\text{old}}}(y_t|x,y_{<t})$:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_{x,y} \Bigl[ \sum_{t=1}^T \min\bigl( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \bigr) \Bigr].
$$

For $\hat{A}_t > 0$ the objective rises with $r_t$ until $1+\epsilon$, then flattens; for $\hat{A}_t < 0$ it falls until $1-\epsilon$, then flattens. This prevents destructive large updates when optimizing multiple epochs on the same rollout batch. The clipping threshold $\epsilon$ is typically $0.2$.

**The KL penalty** serves a *different* purpose: it anchors the policy to the frozen SFT model $\pi^{\text{ref}}$, preserving language quality and preventing reward hacking. The modified per-token reward becomes

$$
r_{\text{total},t} = r_\psi(x,y) - \beta \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi^{\text{ref}}(y_t|x,y_{<t})},
$$

where $\beta \sim 10^{-3}$–$10^{-1}$ (no consensus scaling law). Token-level KL is mathematically equivalent to sequence-level KL for the total return but changes advantage estimation because the penalty is distributed across time steps; most LLM implementations use token-level KL so GAE sees it per step.

**Why both?** Clipping constrains the *per-update* step from $\pi_{\theta_{\text{old}}}$; KL constrains the *cumulative* drift from $\pi^{\text{ref}}$. They are not interchangeable. The full PPO objective adds a value-function loss and optionally a pretraining gradient (PPO-ptx):

$$
L(\theta) = \hat{\mathbb{E}}_{x,y} \Bigl[ \sum_t L_t^{\text{CLIP}}(\theta) - c_1 L_t^{VF}(\theta) \Bigr] + \lambda_{\text{ptx}} \mathbb{E}_{x\sim\mathcal{D}_{\text{pretrain}}}[\log \pi_\theta(x)].
$$

---

## Intuition Check: The Clipping Geometry

The clipping operator is a **hinge** on the policy ratio. When the advantage is positive, we want to increase the probability of the taken action — but only up to a factor of $1+\epsilon$. Beyond that, the gradient is zero: the update "gives up" on pushing further because the ratio estimate is unreliable off-policy. When the advantage is negative, we decrease the probability down to $1-\epsilon$, then stop. This creates a *trust region* without a second-order solver.

```python
import torch

def ppo_clip_loss(ratio, advantage, eps=0.2):
    """Scalar implementation of the PPO clipped surrogate."""
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
    return -torch.min(ratio * advantage, clipped * advantage).mean()

# --- sanity checks ---
# 1. Positive advantage: loss decreases as ratio increases toward 1+eps, then flattens
adv_pos = torch.tensor(1.0)
ratios = torch.linspace(0.5, 1.5, 11)
losses = [ppo_clip_loss(r, adv_pos).item() for r in ratios]
assert losses[5] > losses[6] > losses[7]  # decreasing up to 1.2
assert abs(losses[7] - losses[8]) < 1e-6  # flat after 1.2 (1+eps)

# 2. Negative advantage: loss decreases as ratio decreases toward 1-eps, then flattens
adv_neg = torch.tensor(-1.0)
losses = [ppo_clip_loss(r, adv_neg).item() for r in ratios]
assert losses[5] > losses[4] > losses[3]  # decreasing down to 0.8
assert abs(losses[3] - losses[2]) < 1e-6  # flat after 0.8 (1-eps)

# 3. At ratio=1, loss = -advantage (exact policy gradient)
assert abs(ppo_clip_loss(torch.tensor(1.0), adv_pos).item() + 1.0) < 1e-6
assert abs(ppo_clip_loss(torch.tensor(1.0), adv_neg).item() - 1.0) < 1e-6

print("All checks passed.")
```

---

## Load-Bearing Disagreements

### 1. Reward Translation Sensitivity
The reward model is trained with **relative** feedback (Bradley-Terry loss on pairs), which is invariant to adding a prompt-dependent constant $\delta(x)$ to both responses. PPO optimizes an **absolute** reward, so it *is* sensitive to $\delta(x)$. A reward model that systematically scores certain prompts higher — even if preferences are identical — will steer the policy toward those prompts. This mismatch is a structural flaw; P3O (Pairwise PPO) fixes it by optimizing relative preferences directly and achieves strictly dominant KL-reward frontiers over PPO and DPO.

### 2. GAE Necessity in a Bandit
The original PPO paper advocates GAE for variance reduction in multi-step MDPs. In the LLM bandit setting, the reward is terminal and sparse. Some implementations (e.g., InstructGPT) use a simpler formulation: assign the total reward at sequence end and apply the KL penalty per token, effectively making $\hat{A}_t = r_\psi(x,y) - V_\phi(x)$ for all $t$. The practical impact of per-token vs. sequence-level advantage estimation on alignment quality is not widely reported, but the token-level KL *does* change the GAE recursion because the penalty enters the per-step reward.

---

## Practical Stability Checklist (from "Secrets of RLHF")

| Component | Why it matters |
|-----------|----------------|
| Critic initialized from RM backbone | Warm-starts value estimates; avoids random critic destabilizing early advantages |
| Reward normalization + clipping | Running mean/std + clip to $[-\delta,\delta]$ prevents reward scale drift |
| Small rollout buffer (e.g., 128) + minibatch (e.g., 32) | Limits on-policy staleness; fits GPU memory |
| Global gradient clipping (norm 1.0) | Prevents exploding updates from long sequences |
| Token-level KL in reward | Distributes anchor penalty across GAE steps |
| Early stopping on KL (target 0.01–0.05) | Clipping alone doesn't bound cumulative drift from $\pi^{\text{ref}}$ |
| PPO-ptx (pretraining gradient) | Mitigates alignment tax on general capabilities |

---

## Current Status
PPO remains the default online RL optimizer for industrial RLHF pipelines (OpenAI, Anthropic, Google, Meta) as of 2024, but its share of alignment research is declining as DPO, P3O, and RLAIF gain traction for static datasets; no large-scale ablation proves PPO's superiority over optimally tuned offline methods on the same data and compute.

**Full reference:** See the companion reference article for exhaustive citations, variant tables, hyperparameter logs, and theoretical proofs (global convergence of PPO-Clip, PAC-Bayes generalization bounds, unified KL frameworks).

---
*Full reference (citations, derivations, variants):* [PPO for LLM fine-tuning (RLHF)](../topics/ppo-for-llms.md)
