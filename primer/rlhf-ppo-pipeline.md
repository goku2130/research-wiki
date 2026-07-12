---
title: The RLHF/PPO pipeline
kind: primer
reference: ../topics/rlhf-ppo-pipeline.md
updated: '2026-07-12'
---

# The RLHF/PPO Pipeline: From Preference Learning to Policy Optimization

The RLHF/PPO pipeline — Supervised Fine-Tuning (SFT) → Reward Model (RM) → Proximal Policy Optimization (PPO) — is the canonical three-stage recipe for aligning language models with human intent. Introduced by InstructGPT, it frames alignment as a reinforcement learning problem where a learned reward model provides the signal, and PPO optimizes the policy under a KL constraint to prevent reward hacking and catastrophic forgetting. By the end of this primer you will understand how each stage connects, why the KL penalty is the linchpin, and where the pipeline fractures in practice — knowledge that transfers directly to modern alternatives like DPO, P3O, and GRPO.

---

## Core Mechanism: Preference → Reward → Policy with a KL Leash

The pipeline's logic is a loop: humans compare model outputs → a reward model learns to predict those preferences → the policy is optimized against that reward while staying close to its SFT initialization. The critical insight from Christiano et al. (2017) is that **offline reward learning fails**: once the policy shifts its occupancy distribution, it exploits the reward model's blind spots, producing "bizarre behavior" like avoiding losing in Pong without trying to score. Online data collection — or a strong KL constraint — is non-negotiable.

### Stage 1: Supervised Fine-Tuning (SFT)

SFT adapts a pre-trained base model to the instruction-following distribution by maximizing likelihood on human demonstrations $(x, y)$:

$$
L_{\text{SFT}} = -\sum_i \log P(y_i \mid x, y_{<i}; \theta_{\text{SFT}})
$$

The SFT model becomes both the **initial policy** for PPO and the **reference policy** $\pi^{\text{SFT}}$ for the KL penalty. Practical detail: dropout is disabled across SFT, RM, and PPO to ensure reproducible log-probabilities for the KL term.

### Stage 2: Reward Model (RM) Training

Human labelers rank $K$ completions per prompt, yielding pairwise comparisons $(x, y_w, y_l)$. The RM $r_\phi(x,y)$ outputs a scalar trained with the Bradley–Terry pairwise ranking loss:

$$
\mathcal{L}_R = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma\bigl(r_\phi(x,y_w) - r_\phi(x,y_l)\bigr) \right]
$$

**Crucial nuance**: this loss is *translation-invariant* in $x$ — adding any $\delta(x)$ to both $r_\phi(x,y_w)$ and $r_\phi(x,y_l)$ leaves the loss unchanged. But PPO optimizes *absolute* reward values. This mismatch is the root cause of PPO instability (P3O critique).

The RM is initialized from the SFT model with a random linear head. At inference, the scalar reward is extracted from the **EOS token** position; non-EOS tokens typically carry negative logits. The RM is normalized so a reference set (e.g., SFT summaries) has mean score 0.

### Stage 3: PPO Fine-Tuning

The environment is a contextual bandit: prompt $x$ is sampled, policy $\pi_\phi^{\text{RL}}$ generates $y$, RM returns $r_\theta(x,y)$. The per-token KL penalty from the SFT policy is added to the reward:

$$
R(x,y) = r_\theta(x,y) - \beta \, D_{\text{KL}}\bigl(\pi_\phi^{\text{RL}}(y|x) \,\|\, \pi^{\text{SFT}}(y|x)\bigr)
$$

The objective is $\max_{\pi_\phi} \mathbb{E}_{x\sim\mathcal{D}_{\text{SFT}}, y\sim\pi_\phi}[R(x,y)]$. PPO (Schulman et al. 2017) optimizes this with the **clipped surrogate objective**:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \bigl(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \bigr) \right]
$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ and $\epsilon \approx 0.2$. This creates a *pessimistic lower bound*: when the ratio would improve the objective but lies outside $[1-\epsilon, 1+\epsilon]$, the clip removes the incentive to move further. Combined with a value loss $L^{\text{VF}}$ and entropy bonus $S$, the full loss is:

$$
L_t^{\text{CLIP+VF+S}}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{\text{CLIP}}(\theta) - c_1 L_t^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

**Why the KL penalty works**: it constrains the policy to stay near the SFT distribution, preventing the policy from drifting into regions where the RM is uncalibrated (reward hacking). The coefficient $\beta$ controls the trade-off: N+ uses $\beta=0.05$; 1B models over-optimize (KL 50–85 nats, win rate <20%), while 2.8B+ models are more robust.

---

## Runnable Check: Bradley–Terry Loss and KL-Penalized Reward

```python
import torch
import torch.nn.functional as F

# Bradley–Terry pairwise loss (RM training)
def bt_loss(r_w, r_l):
    # r_w, r_l: (B,) rewards for chosen/rejected completions
    return -F.logsigmoid(r_w - r_l).mean()

# KL-penalized reward (PPO step)
def kl_penalized_reward(r_rm, logp_pi, logp_ref, beta=0.05):
    # r_rm: (B,) RM scores; logp_pi, logp_ref: (B,) log-probs of full sequences
    kl = (logp_pi - logp_ref).mean()          # per-sequence KL ≈ per-token sum
    return r_rm.mean() - beta * kl

# Quick asserts
B = 4
r_w, r_l = torch.tensor([1.0, 0.5, -0.2, 2.0]), torch.tensor([0.0, -0.5, -1.0, 1.0])
assert bt_loss(r_w, r_l).item() > 0

logp_pi, logp_ref = torch.tensor([-5.0, -6.0, -4.0, -7.0]), torch.tensor([-4.5, -5.5, -3.5, -6.5])
r_rm = torch.tensor([0.8, 0.3, 1.2, 0.5])
reward = kl_penalized_reward(r_rm, logp_pi, logp_ref, beta=0.05)
assert reward.item() < r_rm.mean().item()  # KL penalty reduces reward
print("Bradley–Terry loss:", bt_loss(r_w, r_l).item())
print("KL-penalized reward:", reward.item())
```

---

## Load-Bearing Disagreements

1. **PPO vs. DPO: the recipe matters more than the algorithm**. The "Is DPO Superior to PPO?" paper proves $\Pi_{\text{PPO}} \subsetneq \Pi_{\text{DPO}}$ — any PPO solution minimizes the DPO objective, but DPO can find solutions that don't maximize the RL objective (assigning high probability to OOD responses). Yet on CodeLlama-34B, PPO with **advantage normalization, large batch size, and reference-model EMA** beats DPO decisively (CodeContest 22.4% vs 0%, APPS 44.4% vs 34.2%, SafeRLHF safety 99.5% vs 55.4%). Open-source PPO implementations struggle to replicate this; the Interconnects team found PPO "fickle and broken" compared to industry stacks.

2. **Reward translation invariance breaks the RM→PPO link**. The RM is trained on *comparisons* (invariant to $r(x,y) \to r(x,y) + \delta(x)$), but PPO optimizes *absolute* rewards. P3O shows this mismatch causes instability; their pairwise policy gradient is reward-translation-invariant. On the KL–reward frontier: DPO > P3O > PPO > SFT in both KL and proxy reward, but **GPT-4 win rates reverse DPO vs. P3O** (P3O beats DPO 54.6% despite lower reward) because DPO's higher KL produces lower-quality generations. Optimizing the proxy reward too hard hurts quality.

---

## Current Status
The SFT→RM→PPO pipeline remains the production workhorse for industrial RLHF (ChatGPT, Claude, Llama 3), but the research frontier has moved to offline, reward-translation-invariant, or simpler algorithms (DPO, P3O, GRPO) that mitigate PPO's instability, compute cost (4 large models + rollout), and RM–RL mismatch.

## Full Reference
See the comprehensive reference article "The RLHF/PPO pipeline" for exhaustive hyperparameters, N+ implementation details, StackLLaMA parameter-efficient recipes, calibration curves, and the full citation graph.

---
*Full reference (citations, derivations, variants):* [The RLHF/PPO pipeline](../topics/rlhf-ppo-pipeline.md)
