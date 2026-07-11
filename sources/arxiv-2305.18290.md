---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

**Core Problem**
Fine-tuning large language models (LMs) to align with human preferences typically relies on Reinforcement Learning from Human Feedback (RLHF). The standard RLHF pipeline is computationally intensive and often unstable, requiring the separate fitting of a neural reward model followed by reinforcement learning (e.g., PPO) to optimize the policy while constraining divergence from a reference model. This process demands extensive sampling during training, significant hyperparameter tuning, and complex multi-stage optimization.

**Method/Recipe (DPO)**
Direct Preference Optimization (DPO) eliminates the explicit reward model and reinforcement learning loop by directly optimizing the policy using a classification objective. The procedure follows these steps:
1. **Reference Policy Initialization:** Obtain a supervised fine-tuned (SFT) model $\pi_{\text{ref}} = \pi^{\text{SFT}}$. If unavailable, initialize $\pi_{\text{ref}}$ by maximizing the likelihood of preferred completions in the dataset.
2. **Preference Dataset Construction:** Sample pairs of completions $(y_1, y_2)$ from $\pi_{\text{ref}}$ for each prompt $x$, or utilize existing offline datasets of human preferences $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}$, where $y_w$ and $y_l$ denote the preferred and dispreferred responses.
3. **Direct Policy Optimization:** Optimize the parametric policy $\pi_\theta$ to minimize a binary cross-entropy loss derived from the Bradley-Terry preference model. The optimization requires no sampling from $\pi_\theta$ during training, no separate reward network, and minimal hyperparameter tuning.

**Key Formulas**
The DPO objective is derived by reparameterizing the KL-constrained reward maximization objective. The optimal policy for a reward function $r(x,y)$ takes the closed form:
\[
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right),
\]
where $Z(x)$ is a partition function. By inverting this relationship and substituting it into the Bradley-Terry preference probability $p^*(y_1 \succ y_2 \mid x) = \sigma(r^*(x, y_1) - r^*(x, y_2))$, the partition function cancels, yielding the DPO loss:
\[
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right].
\]
The gradient implicitly weights updates by how incorrectly the implicit reward model $\hat{r}_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ ranks the dispreferred completion, preventing model degeneration.

**Key Quantitative Results**
DPO was evaluated on sentiment control, summarization, and dialogue using models up to 6B parameters. Default hyperparameters use $\beta=0.1$, batch size 64, and learning rate $10^{-6}$ ($\beta=0.5$ for summarization). In controlled sentiment generation, DPO’s reward-KL frontier strictly dominates PPO, even outperforming PPO trained with ground-truth rewards (PPO-GT). For TL;DR summarization, DPO achieved a 61% win rate against reference summaries at temperature 0.0, surpassing PPO’s 57% at its optimal temperature, while demonstrating greater robustness to sampling temperature variations. In single-turn dialogue on the Anthropic HH dataset, DPO was the only computationally efficient method to improve over the dataset’s preferred completions, matching the performance of a Best-of-128 baseline. Out-of-distribution evaluation on CNN/DailyMail showed DPO win rates of 0.36 (temp 0) and 0.31 (temp 0.25), compared to PPO’s 0.26 and 0.23. Human evaluations confirmed DPO samples were preferred 58% of the time over PPO samples, with GPT-4 automated judgments correlating strongly with human agreement.

**Stated Limitations**
The authors identify several limitations. First, comprehensive studies are needed to verify how DPO policies generalize out-of-distribution compared to explicit reward-based methods, and whether self-labeling from the DPO policy can effectively utilize unlabeled prompts. Second, the manifestation of reward over-optimization in the direct preference setting remains unclear. Third, experiments were limited to models up to 6B parameters, leaving scaling to larger state-of-the-art architectures as future work. Finally, the authors note that automated GPT-4 evaluations are sensitive to prompt formulation, necessitating further research into eliciting high-quality judgments from automated systems.
