---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-variants
---

**Core Problem**
Aligning large language models (LLMs) with human preferences is hindered by the complexity and instability of traditional Reinforcement Learning from Human Feedback (RLHF). Standard RLHF pipelines require two distinct phases: first, fitting a neural reward model to human preference labels, and second, optimizing the language model policy via reinforcement learning (e.g., PPO) to maximize this reward while constraining KL-divergence from a reference policy. This process demands continuous sampling from the policy during training, incurs high computational costs, requires careful reward normalization, and is sensitive to hyperparameter tuning.

**Method/Recipe**
Direct Preference Optimization (DPO) eliminates explicit reward modeling and RL loops by reformulating the preference learning objective as a direct policy optimization problem. The training recipe proceeds as follows: (1) construct an offline dataset of preference pairs $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$ by sampling completions from a reference policy $\pi_{\text{ref}}$ (typically a supervised fine-tuned model) and annotating human preferences; (2) initialize the target policy $\pi_\theta$ to $\pi_{\text{ref}}$; and (3) optimize $\pi_\theta$ by minimizing a binary cross-entropy loss parameterized directly by the policy. The algorithm implicitly fits a reward function through a change of variables, leveraging the Bradley-Terry preference model to analytically cancel the intractable partition function, thereby enabling stable, supervised-style training without sampling.

**Key Formulas**
The derivation begins with the standard KL-constrained RLHF objective:
\[
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [r_\phi(x,y)] - \beta \mathbb{D}_{\text{KL}}[\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)].
\]
The optimal policy for this objective admits a closed-form solution:
\[
\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp \left( \frac{1}{\beta} r(x, y) \right),
\]
where $Z(x)$ is the partition function. Inverting this relationship yields the reward reparameterization:
\[
r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x).
\]
Substituting this into the Bradley-Terry preference probability cancels $Z(x)$, producing the DPO loss:
\[
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right].
\]
This loss dynamically weights updates by the implicit reward discrepancy, increasing the likelihood of preferred completions while preventing model degeneration.

**Quantitative Results**
Experiments demonstrate DPO's efficiency and performance across multiple benchmarks. In controlled sentiment generation, DPO achieves a strictly superior reward-KL frontier, outperforming both PPO and an oracle PPO variant with ground-truth rewards. For TL;DR summarization, DPO attains a 61% win rate against reference summaries at temperature 0.0, surpassing PPO's 57% at its optimal temperature, while exhibiting greater robustness to sampling temperature variations. In single-turn dialogue on the Anthropic HH dataset, DPO is the only computationally efficient method to improve over dataset-preferred completions, matching the performance of a computationally prohibitive "Best of 128" baseline. Out-of-distribution testing on CNN/DailyMail articles shows DPO maintaining a 0.36 win rate at temperature 0, compared to PPO's 0.26. Default hyperparameters are $\beta=0.1$, batch size 64, and RMSprop with a learning rate of $10^{-6}$, with $\beta=0.5$ applied to summarization. Human evaluation confirms GPT-4 judgments correlate strongly with human annotators.

**Stated Limitations**
The authors identify several limitations. Comprehensive out-of-distribution generalization requires further study, particularly regarding self-labeling with unlabeled prompts. The manifestation of reward over-optimization within the DPO framework remains unclear. Experimental scaling is limited to models up to 6B parameters, leaving larger-scale applications for future work. Additionally, evaluation win rates computed via GPT-4 are sensitive to prompt phrasing, necessitating more robust automated evaluation protocols. Finally, extending DPO to non-text modalities and exploring its integration with preference-based reinforcement learning frameworks are identified as promising directions.
