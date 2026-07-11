---
id: arxiv:2305.18290
type: paper
title: Direct Preference Optimization (Rafailov et al., 2023)
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

**Core Problem**
Large-scale unsupervised language models (LMs) acquire broad world knowledge but lack precise behavioral control. Aligning these models with human preferences typically relies on Reinforcement Learning from Human Feedback (RLHF), which involves a complex, unstable pipeline: first fitting a neural reward model to preference data, then fine-tuning the LM via reinforcement learning (e.g., PPO) to maximize the reward while constraining divergence from a reference policy. This process incurs high computational costs, requires sampling from the LM during training, and demands significant hyperparameter tuning.

**Method and Recipe**
Direct Preference Optimization (DPO) eliminates the explicit reward model and RL loop by deriving a closed-form mapping between reward functions and optimal policies. The algorithm directly optimizes a parameterized policy $\pi_\theta$ using a binary cross-entropy objective. The training recipe proceeds as follows:
1. **Reference Policy Initialization:** Initialize $\pi_{\text{ref}}$ as a supervised fine-tuned (SFT) model. If an SFT model is unavailable, initialize $\pi_{\text{ref}}$ by maximizing the likelihood of preferred completions in the dataset.
2. **Preference Dataset Construction:** Assemble a static dataset $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$ containing prompts $x$, preferred responses $y_w$, and dispreferred responses $y_l$.
3. **Direct Policy Optimization:** Minimize the DPO loss $\mathcal{L}_{\text{DPO}}$ over $\pi_\theta$ using standard supervised learning optimizers (e.g., RMSprop with learning rate $10^{-6}$, batch size 64). The hyperparameter $\beta$ controls the KL-divergence penalty strength (default $\beta=0.1$, or $\beta=0.5$ for summarization). No LM sampling or reward model training occurs during this phase.

**Key Formulas**
DPO leverages the Bradley-Terry preference model, where the probability of preferring $y_1$ over $y_2$ given $x$ is:
\[
p^*(y_1 \succ y_2 \mid x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}.
\]
The optimal policy under a KL-constrained reward maximization objective takes the form:
\[
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right),
\]
where $Z(x)$ is a partition function. By algebraically rearranging this relationship, the reward function can be reparameterized in terms of the policy:
\[
r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).
\]
Substituting this reparameterization into the Bradley-Terry model cancels the partition function, yielding the DPO objective:
\[
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right],
\]
where $\sigma$ is the logistic function. The gradient update implicitly weights examples by how incorrectly the current policy orders the completions, preventing model degeneration.

**Key Quantitative Results**
DPO demonstrates competitive or superior performance across multiple benchmarks without extensive tuning. In controlled sentiment generation, DPO achieves a reward-KL frontier that strictly dominates both PPO and an oracle PPO variant using ground-truth rewards. For TL;DR summarization, DPO achieves a $\approx 61\%$ win rate against reference summaries at temperature $0.0$, outperforming PPO's $57\%$ at its optimal temperature, while maintaining robustness across sampling temperatures. In single-turn dialogue on the Anthropic HH dataset, DPO is the only computationally efficient method to surpass the baseline preferred completions, matching the performance of a computationally expensive Best-of-128 baseline. Under out-of-distribution evaluation on CNN/DailyMail articles, DPO achieves a GPT-4 win rate of $0.36$ at temperature $0$, compared to PPO's $0.26$. Furthermore, human evaluation studies confirm that GPT-4 judgments correlate strongly with human raters, with agreement rates ranging from $65\%$ to $87\%$, comparable to inter-human annotator agreement.

**Stated Limitations**
The authors identify several limitations and directions for future work. First, the out-of-distribution generalization of DPO policies requires more comprehensive study compared to explicit reward-based methods. Second, the manifestation of reward over-optimization within the DPO framework remains unclear. Third, experiments were limited to models up to $6$B parameters, leaving scaling to larger architectures unexplored. Fourth, automated evaluations using GPT-4 are sensitive to prompt phrasing, necessitating better methods for eliciting high-quality judgments. Finally, the framework's extension to other generative modalities and self-labeling techniques with unlabeled prompts is designated for future investigation.
