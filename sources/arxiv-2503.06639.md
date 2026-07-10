---
id: arxiv:2503.06639
type: paper
title: 'Reinforcement Learning with Verifiable Rewards: GRPO''s Effective Loss, Dynamics,
  and Success Amplification'
url: https://arxiv.org/abs/2503.06639
retrieved: '2026-07-10'
maturity: comprehensive
topic: rl-for-reasoning
---

**Core Problem**
The paper investigates Group Relative Policy Optimization (GRPO) when trained with verifiable binary rewards, a paradigm central to recent reasoning-capable LLMs like DeepSeek-R1. It seeks to formally characterize GRPO’s effective loss structure, derive closed-form policy dynamics, and explain its empirical success in amplifying the probability of success (PoS). The analysis systematically contrasts different reward normalization schemes and KL regularization targets to clarify why GRPO outperforms baseline approaches and under what conditions it guarantees monotonic improvement.

**Method/Recipe Step-by-Step**
The authors formalize GRPO iterations by replacing Monte Carlo advantage estimates with analytically derived whitened rewards. The training recipe proceeds as follows: (1) Sample a group of $G$ rollouts from the previous policy $\pi_{t-1}$ for each prompt; (2) Compute the group reward statistics $\mu_t = p_{t-1}$ and $\sigma_t^2 = p_{t-1}(1-p_{t-1})$; (3) Apply mean+variance whitening to the binary reward $R_t \in \{0,1\}$, optionally smoothing the variance with $\epsilon$; (4) Optimize the policy via an importance-sampled objective that combines the whitened advantage with KL regularization; (5) Iterate to $\pi_t$. The paper analyzes four primary variants: standard GRPO with PPO-style clipping and KL regularization to a fixed reference $\pi_{ref}$; Mirror GRPO, which removes clipping and replaces reference regularization with a KL penalty to $\pi_{t-1}$; a mixed variant penalizing both $\pi_{ref}$ and $\pi_{t-1}$; and Dr. GRPO, which applies mean-only normalization instead of mean+variance whitening. Optimization is performed in the probability space, assuming sufficient model capacity to represent any target distribution.

**Key Formulas**
The stabilized whitened advantage is defined as:
$$A_t^{smooth} = \frac{R_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}}$$
The no-clip GRPO objective becomes:
$$\max_{\pi_t} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{t-1}} \left[ \frac{\pi_t(y|x)}{\pi_{t-1}(y|x)} A_t^{smooth} \right] - \beta D_{KL}(\pi_t || \pi_{ref})$$
This formulation reveals GRPO as an adaptive weighted contrastive loss. The resulting PoS recurrence for the standard variant follows Theorem 2:
$$p_t = \sigma\left( \log \frac{p_{t-1}}{1-p_{t-1}} + \log \frac{1-p_{ref}}{p_{ref}} + \log \frac{p_{t-1}}{1-p_{t-1}} \cdot w(p_{t-1}) \right)$$
where $\sigma$ is the sigmoid function and $w(p_{t-1})$ encodes the non-linear weighting induced by reward whitening.

**Key Quantitative Results**
The analysis proves that standard GRPO induces a contrastive loss where correct outcomes are weighted inversely to $p_{t-1}$, while failures are penalized more heavily when success is likely. The PoS sequence converges to a fixed point $p^*$ that strictly exceeds the reference policy’s success rate ($p^* > p_{ref}$), formally demonstrating success amplification. For Mirror GRPO ($\beta_{ref}=0$), the algorithm guarantees monotonic PoS improvement and converges to $p^*=1$ provided $p_{ref} > 0$, or collapses to $p^*=0$ if $p_{ref}=0$. In contrast, the mixed KL variant introduces a Rényi correction term that disrupts monotonicity, while Dr. GRPO’s mean-only normalization yields a trivial fixed point $p^*=p_{t-1}$ or an arithmetic progression in the mirror setting.

**Stated Limitations**
The theoretical tractability requires analyzing no-clip variants, as PPO-style clipping complicates closed-form derivation. The framework assumes the policy space is sufficiently expressive to represent any target distribution in probability space. Fixed points are not guaranteed to be unique, and convergence stability depends on initialization and regularization hyperparameters. The analysis strictly applies to verifiable binary rewards and assumes $p_{t-1} \in (0,1)$ to maintain continuity in the whitening transformation. Furthermore, the theoretical guarantees for monotonic improvement do not extend to mixed KL regularization, where performance depends on the mismatch between reference and current policy success rates.
