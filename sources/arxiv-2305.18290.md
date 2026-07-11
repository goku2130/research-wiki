---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization (DPO): Your Language Model is Secretly a Reward
  Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

**Core Problem**
Aligning large-scale unsupervised language models with human preferences is difficult due to their broad, uncontrolled training objectives. Standard alignment relies on Reinforcement Learning from Human Feedback (RLHF), a complex pipeline that first fits an explicit reward model to human preference data and then fine-tunes the language model via reinforcement learning (e.g., PPO) to maximize the learned reward while constraining KL divergence from a reference policy. This process is computationally expensive, requires continuous policy sampling during training, and suffers from instability and extensive hyperparameter tuning.

**Method/Recipe**
Direct Preference Optimization (DPO) bypasses explicit reward modeling and reinforcement learning by directly optimizing the policy through a simple classification objective. The algorithm proceeds as follows: (1) Initialize a reference policy $\pi_{\text{ref}}$, typically a supervised fine-tuned (SFT) model. (2) Construct an offline dataset $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$ containing prompts and human-labeled preference pairs, where $y_w$ is preferred and $y_l$ is dispreferred. (3) Optimize the target policy $\pi_\theta$ by minimizing a binary cross-entropy loss that directly compares the log-probability ratios of preferred versus dispreferred completions relative to the reference model. (4) Train using standard supervised learning without sampling during the optimization loop or significant hyperparameter search. When an SFT model is unavailable, $\pi_{\text{ref}}$ can be initialized by maximizing the likelihood of preferred completions to mitigate distribution shift between the true reference distribution and the model used for optimization.

**Key Formulas**
The standard RLHF objective maximizes expected reward subject to a KL constraint:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [r_\phi(x,y)] - \beta \mathbb{D}_{\mathrm{KL}}[\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)].
$$

The optimal policy under this objective takes the form $\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$. By inverting this relationship and substituting into the Bradley-Terry preference model, the partition function cancels, yielding the DPO objective:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right) \right],
$$

where $\sigma$ is the logistic function and $\beta$ controls the strength of the implicit KL regularization.

**Quantitative Results**
In controlled sentiment generation, DPO achieves the most efficient reward-KL frontier, strictly dominating PPO and outperforming PPO trained with ground-truth rewards. For TL;DR summarization, DPO achieves a ~61% win rate against reference summaries at temperature 0.0, surpassing PPO's 57% at its optimal temperature, while demonstrating greater robustness to sampling temperature variations. In single-turn dialogue on the Anthropic HH dataset, DPO is the only computationally efficient method to improve over the dataset's preferred completions, matching the performance of a computationally prohibitive "Best of 128" baseline. Under out-of-distribution evaluation on CNN/DailyMail articles, DPO maintains superior win rates (0.36 at temp 0; 0.31 at temp 0.25) compared to PPO (0.26 and 0.23). Human evaluation studies confirm that GPT-4 automated judgments correlate strongly with human preferences, achieving inter-annotator agreement levels.

**Stated Limitations**
The authors note that DPO's out-of-distribution generalization relative to explicit reward learning requires further investigation. The manifestation of reward over-optimization within the DPO framework remains unclear. Experiments were limited to models up to 6B parameters, leaving scaling to larger architectures unexplored. Additionally, the reliance on GPT-4 for automated evaluation introduces prompt sensitivity, necessitating more robust evaluation protocols. Future work must also address how DPO can leverage unlabeled prompts through self-labeling and extend to generative models beyond text.
