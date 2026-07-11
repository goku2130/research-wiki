---
id: arxiv:2409.15360
type: paper
title: Reward-Robust RLHF in LLMs
url: https://arxiv.org/abs/2409.15360
retrieved: '2026-07-10'
maturity: comprehensive
topic: reward-modeling
---

**Core Problem**
Standard Reinforcement Learning from Human Feedback (RLHF) relies on a single Reward Model (RM) to guide policy optimization via Proximal Policy Optimization (PPO). However, RMs are inherently imperfect due to annotator disagreement, overfitting, underfitting, and misalignment with human preferences. This imperfection frequently causes reward hacking, training instability, and suboptimal alignment, prompting the central question: how to perform RM-based RLHF effectively when the reward signal is flawed.

**Method/Recipe**
The authors propose a reward-robust RLHF framework that balances nominal performance with worst-case robustness. The procedure follows these steps: First, construct a Bayesian Reward Model Ensemble (BRME) using a multi-head architecture with parameter sharing. Each head shares a common base feature extractor but outputs two values: the mean and standard deviation of a Gaussian distribution. Second, implement a two-stage training process. Stage one trains a traditional single-head RM using Maximum Likelihood Estimation (MLE) loss. Stage two trains the BRME using a Mean Squared Error (MSE) loss. During inference, all heads evaluate a prompt-response pair; the head with the lowest standard deviation is selected, and its mean output serves as the nominal reward. Third, define the uncertainty set $\mathcal{R}$ using the BRME to capture distributional variance across heads. Fourth, update the policy using a trade-off objective that combines the expected reward under the nominal RM and the minimum expected reward across the uncertainty set, regularized by a KL divergence constraint relative to a reference model.

**Key Formulas**
The standard RLHF objective maximizes:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} [r(x,y)] - \beta \mathbb{D}_{KL}[\pi_\theta(\cdot|x) || \pi_{ref}(\cdot|x)] \quad (4)
$$

The proposed robust framework introduces a trade-off objective:

$$
\max_{\pi_\theta} \alpha \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} [r_{nom}(x,y)] + (1-\alpha) \min_{r \in \mathcal{R}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} [r(x,y)] - \beta \mathbb{D}_{KL}[\pi_\theta(\cdot|x) || \pi_{ref}(\cdot|x)] \quad (7)
$$

where $\alpha \in [0,1]$ controls the performance-robustness trade-off, $r_{nom}$ is the nominal reward, and $\mathcal{R}$ is the uncertainty set.

**Quantitative Results**
Evaluations utilized LLaMA3-8B-Instruct as the actor and a 5-head BRME, trained on HH-RLHF and UltraFeedBack datasets. Experiments ran across 12 benchmarks (e.g., MMLU, GSM8K, ARC) over 800 PPO steps with an experience batch size of 128, distributed across 24 Nvidia H800 GPUs. Annotator consistency experiments revealed approximately 70% agreement between human annotators and domain experts, dropping to approximately 64% for AI annotators. In PPO tuning, varying $\alpha$ from 0 to 1 in 0.2 increments showed that $\alpha=0.2$ and $\alpha=0.4$ consistently outperformed standard RLHF ($\alpha=0$) at 200 steps across most benchmarks. Theoretical analysis via a synthetic toy model (8 prompts, 8 responses, 3-layer MLP policy) demonstrated that standard PPO deviates significantly from the optimal actor, whereas the robust framework recovers the optimal policy.

**Limitations**
The authors note that RM imperfection is fundamental, persisting even with ideal annotators due to insufficient data coverage and training disturbances. A synthetic experiment only accounted for response coverage, acknowledging that real-world prompt coverage gaps would further degrade RM quality. Additionally, the framework warns against using pure worst-case optimization, as it yields overly conservative policies and requires precise uncertainty set modeling, which is difficult in LLM training. The robustness regularization narrows the reward distribution, stabilizing training but potentially constraining peak nominal performance if $\alpha$ is poorly calibrated.
