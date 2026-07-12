---
title: Entropy and exploration in RL fine-tuning
kind: primer
reference: ../topics/entropy-and-exploration.md
updated: '2026-07-12'
---

# Entropy and Exploration in RL Fine-Tuning

**Scaffold.** This primer teaches why entropy collapse is a *structural* failure mode in RL fine-tuning of LLMs — not a hyperparameter bug — and how modern methods (SCOPE-RL, DiPO, REPO/ADAPO, and diversity-preserving objectives) address it. By the end you will understand the two-level collapse mechanism (token + outcome), the temperature–sign asymmetry that makes adaptive control possible, and why "maximal entropy" is the wrong target. This connects to GRPO/DAPO pipelines, verifiable-reward reasoning (RLVR), and the broader exploration literature from contextual MDPs to batch RL regularization.

---

## The Core Mechanism: Why Collapse Is Inevitable Without Intervention

Standard RL maximizes expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$. In discrete-action LLMs this creates an **outcome-frequency multiplier**: trajectories that the policy already assigns high probability get sampled more often, receive reward, and get reinforced *in proportion to their current probability*. The gradient update for a sampled token $a_t$ in a rewarded trajectory is proportional to $\pi_\theta(a_t|s_t)$ — so high-probability tokens get *exponentially* more gradient signal than low-probability ones. This is not a learning-rate artifact; it is baked into the policy-gradient theorem.

In GRPO-style updates the dynamics sharpen further. Each update performs **positive sharpening** (increasing logits of tokens that appeared in rewarded samples) and **negative squeezing** (redistributing probability mass away from tokens that appeared in non-rewarded samples). Once a token's probability drops below the sampling threshold, it effectively vanishes — its gradient becomes zero and it can never recover. This is *irreversible contraction*.

The result is two coupled failure modes:
- **Token-level entropy collapse**: $\mathcal{H}(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log\pi(a|s) \to 0$. The policy becomes nearly deterministic, unable to sample alternative reasoning paths.
- **Outcome-level mode collapse**: The agent converges to a tiny subset $\mathcal{M}$ of reward-supporting solutions with $|\mathcal{M}| \ll |\mathcal{S}_{\text{reward}}|$. Diagnostics show Pass@1 rising while Pass@k plummets, representation rank deficiency, and support shrinkage where incorrect trajectories are compressed by a factor $R_- \approx 0.25\text{--}0.35$.

---

## The Single Most Important Equation: Policy Entropy and Its Dynamics

The policy entropy at state $s$ is

$$
\mathcal{H}(\pi(\cdot|s)) = -\sum_{a \in \mathcal{A}} \pi(a|s) \log \pi(a|s).
$$

In RL fine-tuning we track the *average* entropy $\mathcal{H}(\pi_{\theta_{\text{old}}})$ of the previous-step policy. SCOPE-RL discovered a **temperature–sign asymmetry** that makes this scalar actionable:

- Under **high-temperature** sampling, *positive* (rewarded) samples *increase* entropy while negative samples accelerate collapse.
- Under **low-temperature** sampling, *negative* samples *increase* entropy but cause rapid reward collapse.

This asymmetry means we can *steer* entropy by adjusting temperature *and* filtering which samples contribute to the update. The adaptive temperature rule is

$$
T = \text{clip}\bigl(1 + \mathcal{H}_0 - \mathcal{H}(\pi_{\theta_{\text{old}}}),\; 0.8,\; 1.2\bigr),
$$

where $\mathcal{H}_0$ is a target entropy (empirically $\approx 0.5$). If entropy drifts below target, $T > 1$ flattens the distribution; if it drifts above, $T < 1$ sharpens it. Crucially, SCOPE-RL then draws auxiliary samples at this temperature, *retains only the rewarded ones*, and adds their clipped importance-weighted log-probabilities to the GRPO objective. The full objective is

$$
\mathcal{J}_{\text{SCOPE}}(\theta) = \mathcal{J}_{\text{GRPO}}(\theta) + \alpha \,\mathbb{E}_{q, \{o_i\}\sim\pi_{\theta_{\text{old}}}^T}\Bigl[ \frac{1}{G'}\sum_{i=1}^{G'} \mathbf{1}[R(q,o_i)=1] \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\bigl(r_{i,t}(\theta), \text{clip}(r_{i,t}(\theta),1-\epsilon,1+\epsilon)\bigr) \Bigr],
$$

with $r_{i,t}(\theta) = \pi_\theta(o_{i,t}|q,o_{i,<t}) / \pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})$ and $\alpha = 1/64$ (<2% extra samples). This exploits the asymmetry: high-$T$ positive samples *grow* entropy while the clipping prevents runaway updates.

---

## Runnable Check: Sharpening/Squeezing Dynamics in a Tiny Softmax Policy

The following snippet simulates the positive-sharpening / negative-squeezing loop on a 5-token vocabulary. It shows how a rewarded token's probability grows and an unrewarded token's probability shrinks *irreversibly* once it falls below a sampling floor.

```python
import numpy as np

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()

def grpo_step(logits, rewarded_idx, unrewarded_idx, lr=0.1, floor=1e-4):
    """One GRPO-style update: sharpen rewarded, squeeze unrewarded."""
    pi = softmax(logits)
    # Positive sharpening: increase logit of rewarded token
    logits[rewarded_idx] += lr * (1 - pi[rewarded_idx])
    # Negative squeezing: decrease logit of unrewarded token
    logits[unrewarded_idx] -= lr * pi[unrewarded_idx]
    # Sampling floor: tokens below floor get zero gradient (irreversible)
    pi_new = softmax(logits)
    pi_new[pi_new < floor] = 0.0
    if pi_new.sum() > 0:
        pi_new /= pi_new.sum()
    return pi_new

# Initial uniform logits
logits = np.zeros(5)
rewarded, unrewarded = 0, 1
probs = [softmax(logits).copy()]

for step in range(50):
    pi = grpo_step(logits, rewarded, unrewarded)
    probs.append(pi.copy())

probs = np.array(probs)
# Rewarded token should dominate; unrewarded should vanish
assert probs[-1, rewarded] > 0.95, "Rewarded token didn't sharpen"
assert probs[-1, unrewarded] < 1e-3, "Unrewarded token didn't squeeze"
# Entropy should collapse
entropies = -np.sum(probs * np.log(probs + 1e-12), axis=1)
assert entropies[-1] < 0.2, f"Entropy didn't collapse: {entropies[-1]:.3f}"
print("Collapse demonstrated: rewarded↑, unrewarded↓, entropy→0")
```

Run this to see the structural collapse in ~50 steps — no hyperparameter tuning required.

---

## Modern Solutions: A Taxonomy

### Adaptive Temperature & Perplexity Disentanglement (SCOPE-RL, DiPO)
**SCOPE-RL** (2025) uses the temperature–sign asymmetry above. With $\mathcal{H}_0=0.5$ it lifts Qwen2.5-Math-7B average score from 51.60 → 54.84 and Pass@1024 on AIME24 from 73.3 → 86.7. The entropy–performance curve is **non-monotonic**: $\mathcal{H}_0=0.50$ beats both 0.25 and 0.75, proving maximal entropy is not the goal.

**DiPO** (2026) introduces **Perplexity Space Disentangling (PSD)**: a dynamic threshold $\tau^*$ splits samples into Exploitation Space (low PPL, high correctness) and Exploration Space (high PPL, low correctness). $\tau^*$ minimizes

$$
\frac{1}{|\mathcal{Q}|}\sum_{(r_i,p_i)\in\mathcal{Q}} \bigl| r_i - \mathbb{I}(p_i < \tau) \bigr|
$$

over a cached PPL queue $\mathcal{Q}$. **Bidirectional Reward Reallocation (BRR)** then flips rewards *only in zero-gradient groups*: in hard groups (all 0) within Exploitation Space, the max-PPL sample gets reward 1; in easy groups (all 1) within Exploration Space, the max-PPL sample gets reward 0. The objective $\mathcal{J}_{\text{DiPO}} = \mathcal{J}_{\text{DAPO}}(\theta,\mathcal{R}) + \alpha \mathcal{J}_{\text{DAPO}}(\theta,\mathcal{R}_r)$ with $\alpha=0.1$ is far more robust than entropy bonuses (10× $\alpha$ increase causes only 1.43 pt drop vs. catastrophic collapse for entropy loss).

### Diversity-Preserving Objectives (Beyond Global Entropy Bonuses)
Global entropy bonuses oscillate between collapse and explosion. Five newer frameworks replace them:

| Method | Core Idea | Key Result |
|--------|-----------|------------|
| **Differential Smoothing (DS)** | Penalize high-base-probability *correct* traces; sharpen *incorrect* ones to prevent sharpening onto a limited set | +6.7% Pass@K on math reasoning |
| **Mode Anchored Reward Augmentation (MARA)** | Edit reward landscape so KL target stays flat across all high-reward modes | Near-uniform entropy, Pareto-optimal reward/diversity |
| **DPH-RL** | Replace reverse-KL with mass-covering $f$-divergences (forward-KL, JS) that continuously reference base policy | Prevents Pass@k degradation & catastrophic forgetting |
| **Diversity via Determinants (DvD)** | Maximize volume of behavioral manifold via Gram-matrix determinant | Retains forward/backward behaviors in Humanoid-v2 |
| **Proximal Feature Optimization (PFO)** | Control feature-rank decay to maintain plasticity | Prevents representation collapse |

### Entropy-Preserving Advantage/Clipping (REPO, ADAPO)
**REPO** regulates entropy by modifying the advantage function; **ADAPO** uses adaptive asymmetric clipping. Both maintain trajectory diversity, achieve higher final performance, and — critically — retain "trainability" in sequential learning tasks where standard methods freeze.

### Other Algorithmic Remedies
- **Lp-Reg**: Forward-KL + proxy-KL on filtered low-probability tokens ("reasoning sparks"); +2.66% over standard entropy controls.
- **IPS**: Weight learning signal by $1/p$ to remove self-reinforcing gradient; converges to reward-proportional stationary distribution.
- **UEC-RL**: High-$T$ exploration for hard prompts + replay-based entropy consolidation; 37.9% relative gain over GRPO on Geometry3K.
- **DSDR**: Couples global & local diversity via reward shaping on correct diverse trajectories + token-level entropy.
- **APO**: Support manifold pull + elastic recovery for valid alternatives.

---

## Load-Bearing Caveats (The Two That Matter Most)

1. **Non-monotonic entropy–performance curves are universal.** SCOPE-RL, DiPO, and the diversity-preservation suite all find an *optimal* target entropy ($\mathcal{H}_0 \approx 0.5$ in SCOPE-RL), not maximal entropy. Pushing entropy higher *hurts* reward. This means "entropy bonus = good" is false; *controlled* entropy = good.

2. **Theory lags practice in high dimensions.** All theoretical analyses (SCOPE-RL, DiPO, DPH-RL, etc.) assume tabular softmax, binary rewards, orthogonal gradients, and often only cover $\Delta\mathcal{H} < 0$ regimes. **No convergence guarantees exist for support-preserving regularization in LLM policy spaces.** The batch RL unification result (discount regularization ≡ Bayesian prior ≡ $\epsilon$-greedy planning as transition-matrix smoothing) hints that KL, entropy, and sampling controls may share a common mathematical structure — but this is unproven for LLMs.

---

## Current Status
**Rising, actively researched area with multiple competing frameworks (SCOPE-RL, DiPO, REPO/ADAPO, DS/MARA/DPH-RL/DvD/PFO) and no settled default; large-scale ablations under identical compute budgets are absent.**

**Full reference:** See the linked reference article for 15+ source papers spanning 2019–2026 (contextual MDP bonus trade-offs, VBE first-visit optimism, MEMEC mellowmax, batch RL unification, REDQ/offline fine-tuning, ICQL/CAML architectures).

---
*Full reference (citations, derivations, variants):* [Entropy and exploration in RL fine-tuning](../topics/entropy-and-exploration.md)
