---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

**Core Problem**
Fine-tuning large language models (LLMs) to align with human preferences is difficult because unsupervised pre-training yields broad capabilities without precise behavioral control. Standard Reinforcement Learning from Human Feedback (RLHF) addresses this by first fitting an explicit reward model to human preference data and then optimizing a policy via reinforcement learning (e.g., PPO) to maximize reward while constraining deviation from a reference model via KL-divergence. However, RLHF is computationally expensive, unstable, and requires complex multi-stage training loops with continuous policy sampling.

**Method/Recipe**
Direct Preference Optimization (DPO) eliminates the explicit reward modeling and RL stages by directly optimizing the language model policy using a simple classification loss. The procedure follows these steps: (1) Initialize a reference policy $\pi_{\text{ref}}$, typically a supervised fine-tuned (SFT) model. (2) Construct an offline dataset $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}$ containing prompts and human-labeled preference pairs (preferred $y_w$ and dispreferred $y_l$). (3) Optimize the target policy $\pi_\theta$ by minimizing a binary cross-entropy objective that implicitly fits a reward function parameterized by the policy itself. (4) Train using standard supervised learning gradient descent without sampling from the model or RL hyperparameter tuning. When an SFT model is unavailable, $\pi_{\text{ref}}$ can be initialized by maximizing the likelihood of preferred completions to mitigate distribution shift.

**Key Formulas**
The standard RLHF objective maximizes expected reward subject to a KL constraint:
\[
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [r_\phi(x,y)] - \beta \mathbb{D}_{\text{KL}}[\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)].
\]
The optimal policy under this objective takes the form $\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r(x,y))$. By reparameterizing the reward in terms of the optimal policy and substituting into the Bradley-Terry preference model, the partition function cancels. This yields the DPO loss:
\[
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right].
\]
The gradient of this loss increases the likelihood of preferred completions and decreases dispreferred ones, weighted by how incorrectly the implicit reward model orders the pair.

**Quantitative Results**
DPO was evaluated on sentiment control, TL;DR summarization, and single-turn dialogue using models up to 6B parameters. In controlled sentiment generation, DPO's reward-KL frontier strictly dominates PPO and oracle PPO-GT. For TL;DR summarization, DPO achieves a ~61% win rate against reference summaries at temperature 0.0, outperforming PPO's 57% at its optimal temperature, while demonstrating greater robustness to sampling temperature variations. In single-turn dialogue on the Anthropic HH dataset, DPO is the only computationally efficient method to improve over the dataset's preferred completions, matching the "Best of 128" baseline. PPO failed to outperform the base Pythia-2.8B model. Under out-of-distribution evaluation on CNN/DailyMail news articles, DPO maintains superior win rates (0.36 at temp 0, 0.31 at temp 0.25) compared to PPO (0.26, 0.23). Human evaluation studies confirm that GPT-4 automated judgments correlate strongly with human raters (agreement rates of 65–87%).

**Stated Limitations**
The authors note several limitations. Comprehensive studies on DPO's out-of-distribution generalization relative to explicit reward learning are needed. The manifestation of reward over-optimization within the DPO framework remains unclear. Experiments were limited to models up to 6B parameters, leaving scaling to larger architectures for future work. Additionally, evaluation win rates are sensitive to the specific GPT-4 prompts used, and the method currently relies on fixed offline preference datasets without leveraging unlabeled prompts or self-labeling mechanisms.
