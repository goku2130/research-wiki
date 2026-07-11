---
id: arxiv:2402.14740
type: paper
title: 'ORPO: Odds Ratio Preference Optimization'
url: https://arxiv.org/abs/2402.14740
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-variants
---

**Core Problem**
Reinforcement Learning from Human Feedback (RLHF) for large language models conventionally relies on Proximal Policy Optimization (PPO), which demands loading up to four models simultaneously and requires delicate hyperparameter tuning. The authors argue that PPO’s architectural complexity was originally motivated by high gradient variance and instability in traditional deep reinforcement learning with random initialization. In contrast, RLHF fine-tunes policies that are already warm-started from strong pre-trained and supervised fine-tuned (SFT) checkpoints. This strong initialization concentrates probability mass on a narrow subset of tokens, naturally stabilizing updates. The core problem addressed is whether PPO’s variance-reduction mechanisms and token-level modeling are actually necessary, and whether simpler online reinforcement learning methods can achieve equal or superior alignment performance with significantly reduced computational overhead.

**Method & Recipe Step by Step**
The authors propose abandoning token-level Markov decision process modeling in favor of a bandit formulation where the entire generation is treated as a single action. They introduce REINFORCE (Vanilla Policy Gradient) and its multi-sample extension, REINFORCE Leave-One-Out (RLOO). The RLOO training recipe proceeds as follows:
1. For each prompt $x$, generate $k$ independent completions $y_{(1)}, \dots, y_{(k)}$ from the current policy $\pi_\theta$.
2. Compute the KL-shaped reward $R(y_{(i)}, x)$ for each sample using a reward model and a KL penalty relative to a reference policy.
3. Construct an unbiased, on-the-fly baseline for each sample by averaging the rewards of the remaining $k-1$ samples.
4. Compute the policy gradient using the RLOO estimator, which scales the gradient of the log-probability by the difference between the sample’s reward and its leave-one-out baseline.
5. Update the policy parameters via gradient descent.
This recipe eliminates the need for a separate critic/value network and removes PPO’s clipping mechanisms, as the SFT initialization keeps policy updates inherently stable.

**Key Formulas**
The optimization objective maximizes the expected KL-shaped reward:

$$
R(x,y) = r_{\phi}(x,y) - \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{ref}}(y|x)}
$$

The standard REINFORCE gradient with a baseline $b$ is:

$$
\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(.|x)} [(R(y, x) - b) \nabla_\theta \log \pi_\theta(y|x)]
$$

where a moving average baseline $b_{\mathrm{MA}} = \frac{1}{S} \sum_s R(x^s, y^s)$ can be used for single-sample updates. For multi-sample RLOO, the estimator is:

$$
\frac{1}{k} \sum_{i=1}^k \left[ R(y_{(i)}, x) - \frac{1}{k-1} \sum_{j \neq i} R(y_{(j)}, x) \right] \nabla \log \pi(y_{(i)}|x)
$$

**Key Quantitative Results & Numbers**
Evaluated on the TL;DR Summarize and Anthropic Helpful & Harmless (HH) datasets using Pythia-6.9B and Llama-7B models, RLOO consistently outperforms PPO, DPO, RAFT, and Vanilla PG. RLOO improves win-rates over PPO by 3.2% to 20.3%, achieving final win-rates of 77.9 (TL;DR), 43.7 (HH/Pythia), and 64.1 (HH/Llama) with $k=4$, compared to PPO’s 67.6, 29.2, and 32.0. RLOO demonstrates superior sample efficiency; $\text{RLOO}_{k=2}$ matches or exceeds $\text{RAFT}_{k=4}$ despite using half the online sampling budget. Furthermore, RLOO maintains lower reward variance and exhibits greater robustness to reward noise ($\sigma=3.0, 5.0$) and KL penalty sensitivity ($\beta=0.25, 0.5, 1.0$) than iterative fine-tuning baselines. It also mitigates the alignment tax, preserving language fluency and n-gram diversity better than PPO and Vanilla PG.

**Stated Limitations**
The authors acknowledge several constraints. The study does not investigate reward model over-optimization, where proxy reward trajectories diverge from true human preferences. It also does not explore the LOO baseline within a token-level action framework that models partial sequences or intermediary rewards. Evaluation relies on simulated win-rates using GPT-4 as a proxy rather than direct human preference measurements. Finally, the work is limited to specific reward formulations and does not explore reinforcement learning with traditional NLP metrics such as ROUGE or BLEU.
