---
title: Rejection sampling and Best-of-N
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:1909.08560
- arxiv:2305.18290
- arxiv:2310.15815
- arxiv:2402.10221
- arxiv:2401.10020
- arxiv:2406.07338
open_questions:
- What is the optimal choice of $ N $ for BoN, and how does it depend on the task,
  reward model, and base LM policy?
- Under what conditions does BoN outperform RL-based methods like PPO, and vice versa?
- How can BoN and RSFT be combined with other alignment methods (e.g., DPO, RLAIF)
  to achieve the best of both worlds?
- What are the failure modes of iterative RSFT, and how can they be mitigated?
---

# Rejection Sampling and Best-of-N: Inference-Time Alignment and Fine-Tuning

Rejection sampling and Best-of-N (BoN) are two closely related techniques for aligning language models (LMs) with human preferences at inference time and during fine-tuning. While rejection sampling is a classical statistical method for drawing samples from a target distribution using a proposal distribution, BoN adapts this idea to LMs by generating multiple candidate responses and selecting the highest-scoring one according to a reward model. These methods have gained prominence as computationally efficient alternatives to reinforcement learning (RL)-based alignment, particularly in settings where reward models are available but RL training is prohibitively expensive or unstable.

## Theoretical Foundations

### Rejection Sampling

Rejection sampling is a Monte Carlo method for generating samples from a target distribution $ p(x) $ when direct sampling is intractable, but a proposal distribution $ q(x) $ from which sampling is feasible and a constant $ M $ such that $ p(x) \leq M q(x) $ for all $ x $ are available. The algorithm proceeds as follows:

1. Sample $ x \sim q(x) $.
2. Sample $ u \sim \text{Uniform}(0, 1) $.
3. Accept $ x $ if $ u \leq \frac{p(x)}{M q(x)} $; otherwise, reject and repeat.

In the context of LMs, rejection sampling is applied to align a policy $ \pi_\theta $ with a target distribution defined by a reward model $ r(x, y) $. Here, $ q(x) $ is the base LM policy, and $ p(x) \propto \pi_{\text{ref}}(x) \exp(\beta^{-1} r(x, y)) $, where $ \pi_{\text{ref}} $ is a reference policy (e.g., the initial supervised fine-tuned model) and $ \beta $ controls the strength of the reward signal. The acceptance probability becomes:

$$
P(\text{accept} \mid x, y) = \frac{\pi_{\text{ref}}(y \mid x) \exp(\beta^{-1} r(x, y))}{M q(y \mid x)}.
$$

The constant $ M $ is chosen such that $ M \geq \max_y \frac{\pi_{\text{ref}}(y \mid x) \exp(\beta^{-1} r(x, y))}{q(y \mid x)} $. In practice, $ M $ is often approximated or set heuristically, as exact computation is intractable for high-dimensional output spaces.

### Best-of-N (BoN)

Best-of-N is a discrete approximation to rejection sampling, where $ N $ candidate responses $ \{y_1, \dots, y_N\} $ are sampled from $ q(y \mid x) $, and the response with the highest reward $ y^* = \arg\max_{y_i} r(x, y_i) $ is selected. BoN is equivalent to rejection sampling in the limit as $ N \to \infty $ [source:arxiv:2305.18290], but for finite $ N $, it introduces a bias due to the discrete selection process. The expected reward of BoN is:

$$
\mathbb{E}_{y \sim \text{BoN}_N} [r(x, y)] = \mathbb{E}_{y_1, \dots, y_N \sim q} \left[ \max_{i} r(x, y_i) \right].
$$

BoN is computationally efficient at inference time, as it requires only $ N $ forward passes through the LM and a single reward model evaluation per candidate. However, its performance is inherently limited by the quality of the proposal distribution $ q(y \mid x) $; if $ q $ assigns low probability to high-reward regions, BoN will fail to discover them regardless of $ N $.

## Inference-Time Alignment with BoN

### Practical Implementation

At inference time, BoN is implemented as follows:

1. For a given prompt $ x $, generate $ N $ candidate responses $ \{y_1, \dots, y_N\} $ from the base LM policy $ q(y \mid x) $. This is typically done using temperature sampling or nucleus sampling to ensure diversity.
2. Score each candidate using a reward model $ r(x, y) $, which may be a learned preference model (e.g., trained via Bradley-Terry on human preferences) or a heuristic proxy (e.g., length, fluency, or rule-based metrics).
3. Select the candidate with the highest reward: $ y^* = \arg\max_{y_i} r(x, y_i) $.

The choice of $ N $ trades off computational cost against alignment quality. While empirical studies suggest that BoN performance often improves with larger $ N $, the specific scaling behavior depends on the task and reward model.

### Advantages and Limitations

**Advantages:**
- **Computational Efficiency:** BoN requires no training and can be applied to any pretrained LM with a reward model. This makes it particularly attractive for aligning large models where RL fine-tuning is prohibitively expensive.
- **Interpretability:** The selection process is transparent, as the reward model’s scores can be inspected to understand why a particular candidate was chosen.
- **Robustness to Reward Model Bias:** Unlike RL, which can exploit reward model flaws (e.g., [reward hacking](reward-hacking.md)), BoN is less prone to over-optimization because it does not update the policy’s parameters.

**Limitations:**
- **Proposal Distribution Bottleneck:** BoN’s performance is capped by the quality of the base LM policy $ q(y \mid x) $. If $ q $ assigns low probability to high-reward responses, BoN will fail to discover them, even for large $ N $.
- **Computational Cost at Scale:** Generating $ N $ candidates per prompt can be expensive for large models, particularly if $ N $ is large (e.g., $ N \geq 128 $). This limits BoN’s applicability in latency-sensitive settings.
- **Discrete Selection Bias:** BoN introduces a bias due to the discrete selection process, which can lead to suboptimal performance compared to methods that optimize the policy directly (e.g., RL or DPO).

### Empirical Performance

BoN has been shown to achieve strong performance in alignment tasks, often matching or exceeding the performance of RL-based methods like PPO. For example:
- In TL;DR summarization, **Direct Preference Optimization (DPO)** achieves a win rate of approximately 61% against reference summaries, outperforming PPO’s 57% win rate at its optimal temperature [source:arxiv:2305.18290].
- In single-turn dialogue on the Anthropic HH dataset, DPO matches the performance of a computationally expensive Best-of-128 baseline, demonstrating that BoN can serve as a strong reference point for alignment methods [source:arxiv:2305.18290].

However, BoN’s performance is sensitive to the choice of reward model and sampling strategy. For example:
- Using a poorly calibrated reward model (e.g., one that overfits to length or superficial features) can lead to degenerate behavior, such as selecting overly verbose or sycophantic responses.
- Temperature sampling is critical for generating diverse candidates; greedy decoding or low-temperature sampling can lead to mode collapse, reducing BoN’s effectiveness.

## Rejection-Sampling Fine-Tuning

### Motivation

While BoN is effective at inference time, it does not improve the base LM policy $ q(y \mid x) $. Rejection-sampling fine-tuning (RSFT) addresses this limitation by using BoN to generate high-reward data for supervised fine-tuning (SFT). The goal is to distill the knowledge of the reward model into the policy, thereby improving the proposal distribution for future BoN or RSFT iterations.

### Algorithm

RSFT proceeds as follows:

1. **Data Generation:** For a dataset of prompts $ \{x_i\}_{i=1}^M $, generate $ N $ candidate responses per prompt using the current policy $ q(y \mid x) $. Select the highest-reward candidate $ y_i^* = \arg\max_{y_j} r(x_i, y_j) $ for each prompt.
2. **Supervised Fine-Tuning:** Train the policy $ \pi_\theta $ to maximize the likelihood of the selected responses: $ \mathcal{L}(\theta) = -\mathbb{E}_{(x_i, y_i^*)} [\log \pi_\theta(y_i^* \mid x_i)] $.
3. **Iteration:** Repeat the process using the updated policy $ \pi_\theta $ as the new proposal distribution.

This process can be viewed as a form of [self-improvement](self-improvement-and-self-play.md), where the policy iteratively refines itself using its own generations. RSFT is closely related to [RLAIF](rlaif.md), where the reward model is replaced by an LLM-as-a-judge.

### Theoretical Justification

RSFT can be interpreted as a form of approximate policy iteration. The BoN selection step acts as a policy improvement operator, while the SFT step acts as a policy evaluation operator. Under idealized conditions (e.g., infinite $ N $, perfect reward model, and sufficient data), RSFT converges to the optimal policy under the KL-constrained reward maximization objective:

$$
\pi^* = \arg\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi} [r(x, y)] - \beta \text{KL}(\pi \parallel \pi_{\text{ref}}).
$$

In practice, however, RSFT is limited by:
- The finite $ N $ used in BoN, which introduces bias.
- The quality of the reward model, which may be miscalibrated or overfit.
- The expressivity of the policy class, which may not be able to represent the optimal policy.

### Empirical Performance

RSFT has been shown to achieve strong performance in alignment tasks, often matching or exceeding the performance of RL-based methods like PPO. For example:
- In the [Self-Rewarding Language Models](source:arxiv:2401.10020) framework, iterative RSFT (implemented as DPO on self-generated preference data) improves AlpacaEval 2.0 win rates from 9.94% ($ M_1 $) to 20.44% ($ M_3 $) over three iterations. This demonstrates the potential of RSFT to iteratively refine both the policy and the reward model.
- RSFT scales well with model size. In experiments with models up to 70B parameters, RSFT achieves consistent gains across iterations, though the scaling laws for extended iterative training remain unexplored [source:arxiv:2401.10020].

However, RSFT is not without limitations:
- **Reward Model Overfitting:** If the reward model is overfit to the training data, RSFT can amplify spurious correlations, leading to poor generalization. For example, a reward model that overfits to length may cause RSFT to generate overly verbose responses.
- **Mode Collapse:** RSFT can suffer from mode collapse if the BoN selection step consistently favors a narrow subset of responses. This is particularly problematic if the reward model is poorly calibrated or if the base policy lacks diversity.
- **Computational Cost:** RSFT requires generating $ N $ candidates per prompt, which can be expensive for large models or large datasets. This limits its applicability in resource-constrained settings.

### Comparison to RL-Based Methods

RSFT is often compared to RL-based methods like PPO, which directly optimize the policy using the reward model. Key differences include:
- **Training Stability:** RSFT is more stable than PPO, as it does not require gradient updates through the reward model or careful hyperparameter tuning (e.g., KL penalties, learning rates). This makes RSFT easier to implement and debug.
- **Sample Efficiency:** RSFT is less sample-efficient than PPO, as it requires generating $ N $ candidates per prompt, whereas PPO typically requires only a single sample per prompt. However, RSFT can be more computationally efficient if the reward model is cheap to evaluate.
- **Performance:** RSFT and PPO often achieve comparable performance on alignment tasks, though PPO can sometimes achieve higher rewards due to its ability to explore more effectively. For example, in TL;DR summarization, DPO (which is closely related to RSFT) achieves a win rate of 61%, while PPO achieves 57% [source:arxiv:2305.18290].

## Current Status and Trajectory

### Adoption and Trends

Rejection sampling and BoN are widely used for inference-time alignment, while RSFT and DPO are gaining traction as computationally efficient alternatives to RL-based fine-tuning. Key trends include:
- **Inference-Time Alignment:** BoN is a common technique for aligning LMs at inference time, particularly in settings where RL fine-tuning is impractical (e.g., for very large models or latency-sensitive applications).
- **Fine-Tuning:** RSFT and DPO are increasingly used in research settings (e.g., [source:arxiv:2401.10020], [source:arxiv:2305.18290]) and are likely to see broader adoption as reward models improve and computational costs decrease.
- **Hybrid Methods:** There is growing interest in hybrid methods that combine BoN/RSFT with RL. For example, BoN can be used to generate high-reward data for pretraining, followed by RL fine-tuning to further improve performance. Alternatively, RSFT can be used to warm-start RL training, reducing the number of RL iterations required.

### Trajectory

The trajectory of BoN and RSFT is shaped by several factors:
1. **Reward Model Quality:** The effectiveness of BoN and RSFT depends critically on the quality of the reward model. As reward models improve (e.g., via better training data, architectures, or evaluation protocols), BoN and RSFT are likely to become even more effective. For example, the [Self-Rewarding Language Models](source:arxiv:2401.10020) framework demonstrates that iterative RSFT can improve both the policy and the reward model, leading to superhuman performance.
2. **Computational Efficiency:** The computational cost of BoN and RSFT is a key limitation, particularly for large models or large $ N $. Advances in distributed inference (e.g., [distributed RL training for LLMs](distributed-rl-training.md)) and model parallelism are likely to mitigate this issue, making BoN and RSFT more scalable.
3. **Theoretical Understanding:** The theoretical foundations of BoN and RSFT are still developing. For example, the relationship between BoN and RL is not fully understood, and there is no consensus on the optimal choice of $ N $ or the best way to combine BoN with other alignment methods. As the theoretical understanding improves, BoN and RSFT are likely to become more principled and effective.
4. **Alternative Methods:** BoN and RSFT face competition from other alignment methods, such as [DPO](dpo-and-preference-optimization.md) and [GRPO](grpo.md). DPO, in particular, has gained popularity due to its simplicity and effectiveness, and it is unclear whether BoN/RSFT will remain dominant in the long term. However, BoN and RSFT have unique advantages (e.g., interpretability, robustness to reward model bias) that may ensure their continued relevance.

### Disagreements and Open Questions

There are several areas of disagreement and open questions in the literature:
- **BoN vs. RL:** Some researchers argue that BoN is fundamentally limited by its reliance on the proposal distribution and that RL is necessary for achieving the highest levels of performance. Others counter that BoN can match or exceed RL’s performance when combined with high-quality reward models or iterative refinement. The resolution of this debate likely depends on the specific task and the quality of the reward model.
- **Optimal $ N $:** There is no consensus on the optimal choice of $ N $ for BoN. Empirical studies suggest that larger $ N $ often improves performance, but the specific scaling behavior depends on the task and reward model. Theoretical work is needed to derive principled guidelines for choosing $ N $.
- **Reward Model Overfitting:** The extent to which BoN and RSFT suffer from reward model overfitting is unclear. Some studies suggest that BoN is robust to reward model bias, while others show that RSFT can amplify spurious correlations. More work is needed to understand the conditions under which BoN and RSFT overfit to the reward model.
- **Iterative RSFT:** The [Self-Rewarding Language Models](source:arxiv:2401.10020) framework demonstrates that iterative RSFT can achieve superhuman performance, but it is unclear how far this process can be pushed. For example, does iterative RSFT eventually saturate, or can it continue to improve indefinitely? What are the failure modes of iterative RSFT (e.g., mode collapse, reward hacking)?

## Key Takeaways

- **Rejection Sampling and BoN:** Rejection sampling is a classical Monte Carlo method for sampling from a target distribution using a proposal distribution. Best-of-N (BoN) is a discrete approximation to rejection sampling, where $ N $ candidates are generated from the base LM policy, and the highest-reward candidate is selected.
- **Inference-Time Alignment:** BoN is a computationally efficient method for aligning LMs at inference time. It requires no training and can be applied to any pretrained LM with a reward model. BoN’s performance is limited by the quality of the proposal distribution and the reward model.
- **Rejection-Sampling Fine-Tuning (RSFT):** RSFT uses BoN to generate high-reward data for supervised fine-tuning, iteratively improving the proposal distribution. RSFT is more stable and easier to implement than RL-based methods like PPO, but it is less sample-efficient and can suffer from mode collapse.
- **Empirical Performance:** BoN and RSFT achieve strong performance in alignment tasks, often matching or exceeding the performance of RL-based methods. For example, DPO (a variant of RSFT) achieves a win rate of 61% in TL;DR summarization, while iterative RSFT improves AlpacaEval 2.0 win rates from 9.94% to 20.44% over three iterations.
- **Current Status:** BoN is widely used for inference-time alignment, while RSFT and DPO are gaining traction as computationally efficient alternatives to RL-based fine-tuning. The trajectory of these methods depends on advances in reward model quality, computational efficiency, and theoretical understanding.
- **Open Questions:** Key open questions include the optimal choice of $ N $, the extent to which BoN and RSFT suffer from reward model overfitting, and the long-term potential of iterative RSFT.

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md): RL-based fine-tuning method that directly optimizes the policy using the reward model.
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md): Alternative to RL-based fine-tuning that eliminates the need for a reward model.
- [Reward modeling for LLMs](reward-modeling.md): Training and evaluation of reward models for alignment.
- [RL for LLMs — overview](rl-for-llms-overview.md): Overview of RL-based methods for aligning LMs.
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md): End-to-end pipeline for aligning LMs using RLHF and PPO.
- [RLAIF (RL from AI feedback)](rlaif.md): Alignment method that replaces human feedback with AI-generated feedback.
- [Self-improvement and self-play RL](self-improvement-and-self-play.md): Methods for iteratively improving LMs using self-generated data.
- [Reward hacking in RLHF](reward-hacking.md): Phenomenon where LMs exploit flaws in the reward model to achieve high rewards without satisfying the intended objective.
- [Reward model over-optimization](reward-model-overoptimization.md): Phenomenon where optimizing the policy too aggressively against the reward model leads to poor generalization.
- [Alignment and win-rate evals](alignment-and-winrate-evals.md): Evaluation protocols for measuring alignment quality.
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md): Methods for improving LM performance at test time using additional compute or RL.

---

## References
- [source:arxiv:1909.08560] [Improving Language Models by Recovering from Mistakes (Ziegler et al., 2019)](https://arxiv.org/abs/1909.08560)
- [source:arxiv:2305.18290] [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2310.15815] [Rejection Sampling CoT (Yan et al., 2023)](https://arxiv.org/abs/2310.15815)
- [source:arxiv:2402.10221] [Mathematical Discoveries from Prompt Learning (Chen et al., 2024)](https://arxiv.org/abs/2402.10221)
- [source:arxiv:2401.10020] [Self-Rewarding Language Models (Zhang et al., 2024)](https://arxiv.org/abs/2401.10020)
- [source:arxiv:2406.07338] [Scaling LLM Test Time Compute Optimally (Chen et al., 2024)](https://arxiv.org/abs/2406.07338)
