---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

**Core Problem**
Aligning large-scale unsupervised language models with human preferences is hindered by the complexity and instability of standard Reinforcement Learning from Human Feedback (RLHF). Conventional RLHF pipelines require two distinct stages: first, training an explicit reward model to fit human preference data, and second, optimizing the language model policy via reinforcement learning (e.g., Proximal Policy Optimization) to maximize this reward while constraining KL divergence from a reference policy. This approach is computationally expensive, requires sampling from the model during training, and demands extensive hyperparameter tuning, creating significant barriers to scalable preference alignment.

**Method: Direct Preference Optimization (DPO)**
DPO circumvents explicit reward modeling and RL loops by deriving a closed-form mapping between reward functions and optimal policies, enabling direct policy optimization via a simple classification loss. The training recipe proceeds as follows:
1. Construct an offline dataset of human preferences $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$, where $y_w$ and $y_l$ denote the preferred and dispreferred completions for prompt $x$.
2. Establish a reference policy $\pi_{\mathrm{ref}}$, typically a supervised fine-tuned (SFT) model. If an SFT model is unavailable, initialize $\pi_{\mathrm{ref}}$ by maximizing the likelihood of preferred completions to mitigate distribution shift.
3. Optimize the target policy $\pi_\theta$ by minimizing a binary cross-entropy loss that compares the log-probability ratios of preferred versus dispreferred responses relative to $\pi_{\mathrm{ref}}$, scaled by a temperature parameter $\beta$. This requires only standard supervised fine-tuning, eliminating sampling loops and actor-critic algorithms.

**Key Formulas**
The derivation assumes the Bradley-Terry preference model, which defines the probability of human preference as:

$$
p^*(y_1 \succ y_2 \mid x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}.
$$

Under a KL-constrained reward maximization objective, the optimal policy takes the form:

$$
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right),
$$

where $Z(x)$ is a partition function. By algebraically rearranging this to express the reward in terms of the policy, the partition function cancels out when substituted into the preference model. This yields the DPO objective:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{w} \mid x)}{\pi_{\mathrm{ref}}(y_{w} \mid x)} - \beta \log \frac{\pi_{\theta}(y_{l} \mid x)}{\pi_{\mathrm{ref}}(y_{l} \mid x)} \right) \right],
$$

where $\sigma$ is the logistic function. The gradient of this loss implicitly weights updates by how incorrectly the model's implicit reward orders the completions, preventing policy degeneration that occurs with naive probability ratio objectives.

**Quantitative Results**
Experiments across sentiment modulation, summarization, and dialogue demonstrate DPO's efficacy. In controlled sentiment generation, DPO's reward-KL frontier strictly dominates PPO, even when PPO accesses ground-truth rewards. On the Reddit TL;DR summarization dataset, DPO achieves a ~61% win rate against reference summaries at temperature 0.0, surpassing PPO's 57% at its optimal temperature, while exhibiting greater robustness to sampling temperature variations. For single-turn dialogue on the Anthropic Helpful and Harmless dataset, DPO is the only computationally efficient method to improve over the dataset's preferred completions, matching the performance of a "Best of 128" baseline. Out-of-distribution evaluation on CNN/DailyMail news articles shows DPO maintaining a 0.36 win rate at temperature 0 and 0.31 at 0.25, outperforming PPO's 0.26 and 0.23, respectively. Human evaluation studies confirm that GPT-4 automated judgments correlate strongly with human raters, with agreement rates ranging from 65% to 87%.

**Stated Limitations**
The authors identify several limitations. First, the out-of-distribution generalization of DPO compared to explicit reward learning requires more comprehensive study, particularly regarding the use of unlabeled prompts. Second, the manifestation of reward over-optimization within the DPO framework remains unclear. Third, all experiments were conducted on models up to 6B parameters, leaving scaling to larger architectures as future work. Finally, the authors acknowledge that automated evaluation metrics (e.g., GPT-4 win rates) are sensitive to prompt design, necessitating further research into robust evaluation protocols and the extension of DPO to non-text modalities.
