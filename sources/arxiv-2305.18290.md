---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

Direct Preference Optimization (DPO) addresses the instability and computational overhead of aligning large language models with human preferences. Traditional Reinforcement Learning from Human Feedback (RLHF) requires two distinct stages: first, fitting an explicit reward model to human preference data, and second, fine-tuning the language model via reinforcement learning (e.g., PPO) to maximize reward while constraining KL divergence from a reference policy. This pipeline is complex, prone to instability, and necessitates sampling from the language model during training. DPO eliminates the explicit reward modeling and reinforcement learning loops by directly optimizing the policy through a single classification objective.

The DPO training recipe proceeds in three steps. First, an offline dataset of human preferences $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$ is constructed, where $y_w$ and $y_l$ represent the preferred and dispreferred completions for a given prompt $x$. Second, a reference policy $\pi_{\text{ref}}$ is established, typically via supervised fine-tuning on high-quality demonstrations. Third, the target language model $\pi_\theta$ is optimized by minimizing the DPO loss, which implicitly fits a reward function through a change of variables that maps the optimal RLHF policy directly to the language model parameters. This approach requires no actor-critic loops, no reward model training, and no in-loop sampling.

The mathematical derivation begins with the standard KL-constrained reward maximization objective:
\[
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [r_\phi(x,y)] - \beta \mathbb{D}_{\text{KL}}[\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)].
\]
The optimal policy under this objective is analytically tractable: $\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r(x,y))$. By inverting this relationship to express the reward as $r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$ and substituting it into the Bradley-Terry preference model, the partition function $Z(x)$ cancels. This yields the final DPO objective:
\[
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right].
\]
The gradient of this loss increases the likelihood of preferred completions and decreases dispreferred ones, weighted by how incorrectly the implicit reward model orders the pair.

Empirical evaluations demonstrate that DPO matches or exceeds PPO-based RLHF across multiple benchmarks. In controlled sentiment generation, DPO produces a strictly superior reward-KL frontier, achieving higher rewards for all KL divergence levels compared to PPO. For TL;DR summarization, DPO achieves a 61% win rate against reference summaries at temperature 0.0, outperforming PPO’s 57% at its optimal temperature, while demonstrating greater robustness to sampling temperature variations. In single-turn dialogue on the Anthropic Helpful and Harmless dataset, DPO is the only computationally efficient method to improve over the dataset's preferred completions, matching the performance of a "Best of 128" baseline that requires extensive sampling. Under out-of-distribution conditions on CNN/DailyMail news articles, DPO maintains a 0.36 win rate at temperature 0 against ground-truth summaries, surpassing PPO’s 0.26. Automated evaluations using GPT-4 correlate strongly with human judgments, with agreement rates ranging from 65% to 87%.

The authors identify several limitations. Comprehensive out-of-distribution generalization studies are required, as initial results suggest DPO generalizes similarly to PPO but lacks the ability to leverage additional unlabeled prompts during training. The manifestation of reward over-optimization within the DPO framework remains unclear. Evaluations were restricted to models up to 6B parameters, leaving scaling to larger architectures for future work. Additionally, GPT-4-based win rates are sensitive to prompt formulation, necessitating careful evaluation design.
