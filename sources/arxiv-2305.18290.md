---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

**Core Problem**
Aligning large language models (LMs) with human preferences traditionally relies on Reinforcement Learning from Human Feedback (RLHF). The standard RLHF pipeline is computationally expensive, unstable, and architecturally complex. It requires two distinct phases: first, training a separate neural reward model to fit human preference data, and second, optimizing the LM policy via reinforcement learning (e.g., PPO) to maximize the learned reward while penalizing divergence from a reference policy. This process demands extensive LM sampling during training, significant hyperparameter tuning, and careful reward normalization to prevent mode collapse and distributional shift.

**Method: Direct Preference Optimization (DPO)**
DPO bypasses explicit reward modeling and RL loops by deriving a closed-form mapping between reward functions and optimal policies, enabling direct policy optimization via a binary classification objective. The algorithmic recipe is:
1. Initialize a reference policy $\pi_{\text{ref}}$, typically a supervised fine-tuned (SFT) model.
2. Construct an offline dataset of human preferences $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$, where $y_w$ and $y_l$ denote the preferred and dispreferred completions for prompt $x$.
3. Optimize the policy parameters $\theta$ by minimizing the DPO loss, which treats preference prediction as a maximum likelihood problem over the policy directly.
4. Train using standard supervised learning updates without sampling from the LM during fine-tuning or requiring reward model normalization.

**Key Formulas**
The derivation starts from the standard KL-constrained reward maximization objective:
$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [r(x, y)] - \beta \mathbb{D}_{\mathrm{KL}}[\pi_\theta(y|x) \mid\mid \pi_{\mathrm{ref}}(y|x)].
$$
Solving for the optimal policy yields $\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$. Inverting this relationship reparameterizes the reward as $r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)$. Substituting this into the Bradley-Terry preference model $p^*(y_1 \succ y_2|x) = \sigma(r^*(x,y_1) - r^*(x,y_2))$ cancels the partition function $Z(x)$, yielding the DPO objective:
$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right) \right].
$$
The gradient implicitly weights updates by how incorrectly the current policy orders the completions, preventing degeneration that occurs with naive probability ratio objectives.

**Key Quantitative Results**
Experiments across sentiment control, summarization, and dialogue demonstrate DPO's efficacy. In controlled sentiment generation, DPO's reward-KL frontier strictly dominates PPO and oracle PPO (PPO-GT), achieving higher rewards at equivalent KL constraints. For TL;DR summarization, DPO achieves a 61% win rate against reference summaries at temperature 0.0, surpassing PPO's 57% at its optimal temperature, while exhibiting greater robustness to sampling temperature variations. In single-turn dialogue on the Anthropic Helpful and Harmless dataset, DPO is the only computationally efficient method to improve over the baseline preferred completions, matching the performance of a Best-of-128 baseline. Out-of-distribution evaluation on CNN/DailyMail articles shows DPO achieving a 0.36 GPT-4 win rate at temperature 0, compared to PPO's 0.26. Human evaluation studies confirm that GPT-4 judgments correlate strongly with human annotators, with agreement rates ranging from 65% to 87%.

**Stated Limitations**
The authors identify several limitations and directions for future work. First, comprehensive studies are needed to rigorously evaluate DPO's out-of-distribution generalization compared to explicit reward modeling. Second, the manifestation of reward over-optimization within the DPO framework remains unclear, and slight performance decreases observed during training may indicate this phenomenon. Third, the current evaluation is limited to models up to 6B parameters, leaving scaling to larger state-of-the-art architectures unexplored. Fourth, the reliance on GPT-4 for automated evaluation introduces prompt-dependent biases, necessitating further research into robust automated preference metrics. Finally, the current framework operates on offline preference datasets; leveraging self-labeling from the DPO policy to utilize unlabeled prompts requires additional investigation.
