---
title: RL for reasoning models
kind: primer
reference: ../topics/rl-for-reasoning.md
updated: '2026-07-12'
---

# Reinforcement Learning for Reasoning Models: A Primer

**What this teaches.** By the end, you will understand how *reinforcement learning with verifiable rewards* (RLVR) replaced learned reward models with executable verifiers to become the dominant paradigm for training large reasoning models (LRMs). You will see the core theoretical mechanism — the **logic prior** — that lets outcome-only rewards implicitly upweight correct reasoning chains, the **GRPO** algorithm that makes thousand-step RL tractable, and the **multi-stage recipe** (cold-start SFT → large-scale RL → rejection sampling → final RL) that produced DeepSeek-R1 and its peers. You will also grasp the two load-bearing disagreements: whether cold-start SFT is strictly necessary for reasoning emergence, and whether distilled models genuinely acquire new reasoning or merely improve sampling efficiency. This connects to policy-gradient theory, inference-time scaling, and the broader shift from human-preference RLHF to verifiable-domain RL.

---

## The Core Mechanism: Verifiable Rewards and the Logic Prior

Classical RLHF trains a *learned* reward model on human preferences, then optimizes the policy against it. This creates two problems: **reward-model over-optimization** (the policy exploits the reward model's errors) and **subjectivity** (preferences are noisy and hard to scale). RLVR sidesteps both by replacing the learned reward model with a **deterministic verifier** that checks the final answer:

- **Math**: $R(y) = \mathbb{1}[\text{extracted\_answer}(y) = \text{ground\_truth}]$
- **Code**: $R(y) = \mathbb{1}[\text{all\_unit\_tests\_pass}(y)]$ (or a proportional score)

The response $y = (c, a)$ splits into a chain-of-thought $c$ and an answer $a$. Crucially, the verifier *only sees $a$* — it has no access to the reasoning $c$. So why does the policy learn to produce *better reasoning*?

The answer is the **logic prior**. Define indicator variables $\mathcal{I}_{\mathrm{CoT}}(c)=1$ if the CoT is logically correct, and $\mathcal{I}_{\mathrm{Ans}}(a)=1$ if the answer is correct. The logic prior states:

$$
P(\mathcal{I}_{\mathrm{Ans}}=1 \mid \mathcal{I}_{\mathrm{CoT}}=1) = \alpha \;>\; \beta = P(\mathcal{I}_{\mathrm{Ans}}=1 \mid \mathcal{I}_{\mathrm{CoT}}=0)
$$

Correct reasoning chains are more likely to yield correct answers than incorrect ones. This inequality is the *theoretical linchpin*: even though the reward observes only answer correctness, the policy gradient receives a stronger positive signal from trajectories with correct reasoning. Over many samples, the optimizer "discovers" that correct CoTs are the reliable path to reward.

**Common confusion pre-empted.** The logic prior does *not* require the model to already reason well. It only requires that, *in the space of possible CoTs*, correct ones correlate with correct answers. A base model sampling randomly will still produce some correct CoTs by chance; the verifier amplifies those trajectories, and the policy gradient shifts probability mass toward them. This is why RLVR can start from a base model without any CoT supervision (as DeepSeek-R1-Zero demonstrated).

---

## The Algorithm: GRPO — Critic-Free, Group-Normalized Advantages

Policy optimization needs an advantage estimator. PPO uses a learned value network (critic), which adds memory and compute overhead — problematic when training for **thousands of RL steps** (DeepSeek-R1 uses ~1000s vs. 100s for earlier recipes). **Group Relative Policy Optimization (GRPO)** eliminates the critic entirely.

For a prompt $q$, sample a group of $G$ responses $\{y_i\}_{i=1}^G$ from the current policy. Compute their rewards $R(y_i)$, then estimate the advantage by **Monte Carlo normalization within the group**:

$$
\hat{A}(y_i) = \frac{R(y_i) - \mu_{\mathbf{Y}}}{\sigma_{\mathbf{Y}}}, \quad
\mu_{\mathbf{Y}} = \frac{1}{G}\sum_{j=1}^G R(y_j), \quad
\sigma_{\mathbf{Y}} = \sqrt{\frac{1}{G}\sum_{j=1}^G (R(y_j) - \mu_{\mathbf{Y}})^2}
$$

The policy gradient update becomes:

$$
\nabla_\theta J(\theta) \approx \frac{1}{G}\sum_{i=1}^G \hat{A}(y_i) \nabla_\theta \log \pi_\theta(y_i \mid q)
$$

The full GRPO objective adds PPO-style clipping and a KL penalty against a reference model (typically the SFT checkpoint):

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^G \left( \min\left( r_i A_i, \text{clip}(r_i, 1-\varepsilon, 1+\varepsilon) A_i \right) - \beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right) \right]
$$

where $r_i = \pi_\theta(y_i|q) / \pi_{\theta_{\text{old}}}(y_i|q)$ and the KL estimator is $\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$.

**Why this works for reasoning.** Group normalization centers advantages around the *current policy's performance on this specific prompt*. A response that beats the group average gets a positive advantage, even if its absolute reward is low. This auto-calibrates the learning signal across prompts of varying difficulty — essential when the reward is sparse (0/1) and the policy improves dramatically over training.

---

## The Canonical Recipe: DeepSeek-R1's Four-Stage Pipeline

OpenAI's o1/o3 proved the paradigm but kept the recipe closed. DeepSeek-R1 (2025) published a replicable pipeline that the field has converged on:

| Stage | What happens | Key details |
|-------|--------------|-------------|
| **1. Cold-start SFT** | Fine-tune base model on ~few thousand filtered long-CoT completions | From an intermediate R1-Zero model; few-shot prompting + reflection/verification prompts + human annotation; enforces readable formatting (e.g., `<think>` tags) |
| **2. Large-scale RL** | GRPO on verifiable reasoning problems "until convergence" | **1000s of RL steps**; reward = accuracy bonus + format reward (`

---
*Full reference (citations, derivations, variants):* [RL for reasoning models](../topics/rl-for-reasoning.md)
