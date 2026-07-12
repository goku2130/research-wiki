---
title: RL for LLMs — overview
kind: primer
reference: ../topics/rl-for-llms-overview.md
updated: '2026-07-12'
---

# Reinforcement Learning for LLMs: From RLHF to Verifiable-Reward Reasoning

**Scaffold.** This primer teaches the architecture of modern LLM alignment and reasoning through reinforcement learning. You will understand how the field moved from the canonical three-stage RLHF pipeline (SFT → reward model → PPO) to offline preference methods that eliminate the reward model, to verifiable-reward RL (RLVR) that replaces learned rewards with ground-truth checkers for reasoning, and finally to offline multi-step methods that learn from static datasets. The connecting thread is a progressive relaxation of infrastructure demands: fewer model copies, less on-policy rollout, and ultimately no human preference annotation at all.

---

## 1. The Canonical Mechanism: RLHF as KL-Constrained Policy Optimization

The foundational insight is that aligning an LLM is a **constrained optimization problem**: maximize expected reward while staying close to a reference policy (the SFT model) to avoid collapse. The canonical pipeline implements this in three stages:

1. **SFT** on human demonstrations $\mathcal{D}_{\text{SFT}}$ yields $\pi_{\text{SFT}}$.
2. **Reward Model (RM)** trained on pairwise preferences $\mathcal{D}_{\text{pref}} = \{(x, y_w, y_l)\}$ via Bradley–Terry:

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}\Bigl[\log\sigma\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr)\Bigr].
$$

3. **PPO** optimizes $\pi_\theta$ against the frozen RM with a KL penalty toward $\pi_{\text{ref}} = \pi_{\text{SFT}}$:

$$
\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{PPO}}(\theta) - \beta\,\mathcal{D}_{\text{KL}}\bigl(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\bigr),
   \quad
   \mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t\Bigl[\min\bigl(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\bigr)\Bigr].
$$

**Why this works.** The KL penalty is not a heuristic—it is the Lagrange multiplier of the constraint $\mathcal{D}_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}}) \le \delta$. The clipped surrogate prevents destructive policy updates. InstructGPT showed this recipe at scale: 175B InstructGPT beat 175B GPT-3 **85±3%** of the time, and the **1.3B InstructGPT beat 175B GPT-3**. The alignment tax on standard benchmarks was mitigated by the PPO-ptx objective that mixes in a pretraining log-likelihood term.

**Common confusion pre-empted.** The RM is *not* the reward function used in PPO; it is a *learned proxy* for human preference. PPO optimizes the policy against this frozen proxy, which creates a distributional shift: the policy visits regions where the RM is unreliable (reward hacking). The KL penalty is the primary brake on this shift.

---

## 2. The Offline Pivot: Direct Preference Optimization

DPO's key insight: the optimal KL-constrained policy has a closed form,

$$
\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\exp\bigl(\tfrac{1}{\beta}r(x,y)\bigr),
$$

which implies an **implicit reward** $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$. Substituting into the Bradley–Terry model cancels the partition function $Z(x)$, yielding a loss that depends *only* on the policy and reference model:

$$
\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)}\Bigl[\log\sigma\Bigl(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigr)\Bigr].
$$

**Why this matters.** DPO eliminates the RM and the PPO rollout loop entirely. It reduces the memory footprint from 4 models ($\pi,\pi_{\text{ref}},r_\phi,V_\psi$) to 2 ($\pi,\pi_{\text{ref}}$) and converts an online RL problem into a supervised classification problem on preference pairs. On summarization and dialogue, DPO matches or exceeds PPO (TL;DR: 61% vs 57% win rate; Anthropic HH: only efficient method improving over preferred completions).

**The catch.** DPO's implicit reward can explode on noisy labels ("catastrophic overfitting"), and it lacks PPO's on-policy exploration. The survey notes an **open gap in robustness to label noise** and a tendency toward length bias. The "Is RLHF Dead?" study later showed that **on-policy DPO** (generating fresh pairs from the current policy, then applying the DPO loss) outperforms both pure online PPO and pure offline DPO, blending exploration with the stable contrastive objective.

---

## 3. Verifiable-Reward RL (RLVR): Reasoning Without Human Labels

For reasoning tasks (math, code, logic), human preference annotation is expensive and noisy. RLVR replaces the learned RM with **verifiable rewards**—automated ground-truth checks (unit tests, proof verifiers, symbolic executors). Three paradigms:

| Paradigm | Reward Signal | Core Limitation |
|----------|---------------|-----------------|
| **Outcome-Based (OB-RL)** | Sparse terminal $R(\tau)\in\{0,1\}$ | High-variance gradients; credit assignment fails if final answer wrong; rewards lucky guesses |
| **Process-Supervised (CoT-RO)** | Dense step-wise $r_t$ from a Process Reward Model | PRM training needs expensive step-level annotation; "reasoning hacking" (confident tone ≠ validity) |
| **Verifier-Guided (V-RL)** | External verifier $V_\phi$ (learned/heuristic/tool) | Adversarial Goodharting: policy attacks verifier; compute doubles |

**GRPO (Group Relative Policy Optimization)** is the workhorse for RLVR. It removes the critic by normalizing rewards within a group of $G$ samples per prompt:

$$
\hat{A}_i^{\text{GR}}=\frac{r(a_i)-\mu}{\sigma},\qquad
\mathcal{L}^{\text{GRPO}}(\theta)=\mathbb{E}\Bigl[\frac1G\sum_{i=1}^G\min\bigl(r_i(\theta)\hat{A}_i^{\text{GR}},\text{clip}(r_i(\theta),1-\epsilon,1+\epsilon)\hat{A}_i^{\text{GR}}\bigr)\Bigr].
$$

Memory drops to **2 models** ($\pi,\pi_{\text{ref}}$), but the compute budget shifts: stable $\sigma$ estimation requires $G\ge 64$, and the method fails catastrophically when all rewards in a group are identical ($\sigma\to 0$).

**DeepSeekMath** demonstrated GRPO at scale: continued pretraining on a 120B-token math corpus (56% math, 20% code) from DeepSeek-Coder-Base 7B, then SFT on 776K CoT/PoT examples, then GRPO with an unbiased KL estimator. Results: **51.7% MATH, 88.2% GSM8K** (60.9% MATH with self-consistency@64), beating Minerva 540B's 33.6% MATH. Code pretraining benefited both tool-use and non-tool-use math; arXiv pretraining showed no notable gain.

**The fundamental debate.** "It remains debated whether RLVR truly expands LLM reasoning capabilities beyond pre-training or merely amplifies high-reward outputs already present in the base model's distribution." Some evidence suggests RLVR improves *sampling efficiency* of correct paths that already exist in the base model's support.

---

## 4. Offline Multi-Step RL: Learning from Static Trajectories

When on-policy rollouts are too expensive, offline methods learn from fixed datasets of reasoning trajectories. **OREO** enforces a telescoped soft Bellman equation:

$$
V_\phi(s_t)=R_t-\beta\sum_{i=t}^{T-1}\log\frac{\pi_\theta(a_i|s_i)}{\pi_{\text{ref}}(a_i|s_i)},\quad
\mathcal{L}_V(\phi)=\frac1T\sum_t\Bigl(V_\phi(s_t)-R_t+\beta\sum_{i=t}^{T-1}\log\frac{\pi_\theta(a_i|s_i)}{\pi_{\text{ref}}(a_i|s_i)}\Bigr)^2.
$$

On Qwen-2.5-Math 1.5B: **+5.2% GSM8K, +10.5% MATH** vs SFT; learned $V_\phi$ enables test-time beam search ($B=7$: **+11.4% GSM8K, +17.9% MATH**). **ILQL/VerifierQ** apply Conservative Q-Learning regularization but are limited by offline dataset support and overestimation bias ("delusional value estimation" for OOD tokens). **DSPy** frames pipeline optimization as compilation: signatures, modules, teleprompters; GSM8K GPT-3.5: 24% → **88.3%** (CoT ensemble). These methods are strong at small scale but **not validated at frontier scale (>100B)**.

---

## 5. Runnable Check: GRPO Advantage Normalization

The GRPO advantage calculation is the mechanism that replaces the critic. This snippet verifies the normalization and the singularity failure mode.

```python
import torch

def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute GRPO group-relative advantages. rewards shape: (G,)"""
    mu = rewards.mean()
    sigma = rewards.std(unbiased=False) + eps  # population std + guard
    return (rewards - mu) / sigma

# Stable case: diverse rewards
rewards_diverse = torch.tensor([0.0, 0.5, 1.0, 0.2, 0.8])
adv = grpo_advantages(rewards_diverse)
assert adv.shape == (5,)
assert abs(adv.mean().item()) < 1e-6
assert abs(adv.std(unbiased=False).item() - 1.0) < 1e-6

# Singularity case: homogeneous rewards -> sigma ~ 0 -> huge advantages
rewards_homog = torch.tensor([1.0, 1.0, 1.0, 1.0])
adv_homog = grpo_advantages(rewards_homog)
# With eps=1e-8, advantages explode to ~1e8; in practice this destabilizes training
assert adv_homog.abs().max() > 1e7
print("GRPO advantages: diverse OK, homogeneous -> singularity (as predicted).")
```

---

## 6. Load-Bearing Disagreements

1. **DPO vs. PPO robustness.** The DPO paper reports DPO *dominates* PPO on the reward–KL frontier and shows greater robustness to sampling temperature. The technical survey [2507.04136] counters that DPO has an **open gap in robustness to label noise** and is susceptible to length bias and implicit reward explosion. The "Is RLHF Dead?" study [2404.14367] resolves this partially: **on-policy DPO** (fresh on-policy pairs + DPO loss) beats both, suggesting the gap stems from *offline* data limitations, not the loss itself.

2. **RLVR: capability expansion vs. sampling efficiency.** The RLVR survey [2509.16679] states it "remains debated whether RLVR truly expands LLM reasoning capabilities beyond pre-training or merely amplifies high-reward outputs already present in the base model's distribution." If the latter, RLVR hits a ceiling defined by the base model's support; if the former, it is a genuine capability ladder. No consensus exists.

---

**Current status.** RLHF (PPO) remains the reference alignment pipeline but is fading for frontier reasoning due to 4-model memory bottleneck and instability; RLVR (GRPO + verifiable rewards) is the dominant paradigm for reasoning; offline PO (DPO, ORPO, DRO) is default for preference tuning with 2-model footprint; offline multi-step RL (OREO, ILQL) shows promise at small scale but is unvalidated at frontier scale.

**Full reference.** See the comprehensive survey "RL for LLMs — overview" for all equations, tables, and 20+ citations spanning RLHF, RLAIF, DPO/DRO/ORPO/UNA, RLVR (OB-RL, CoT-RO, V-RL, GRPO, DeepSeekMath, OPRO), offline RL (OREO, ILQL, VerifierQ, DSPy), and the online–offline trade-off analysis.

---
*Full reference (citations, derivations, variants):* [RL for LLMs — overview](../topics/rl-for-llms-overview.md)
