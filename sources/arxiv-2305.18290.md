---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization (DPO): Direct Preference Optimization'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

**Core Problem**
Aligning large-scale unsupervised language models (LMs) with human preferences remains challenging due to the instability and computational complexity of existing Reinforcement Learning from Human Feedback (RLHF) pipelines. Standard RLHF requires two distinct phases: first, fitting an explicit reward model to human preference data, and second, fine-tuning the LM via reinforcement learning (e.g., PPO) to maximize this reward while constraining KL divergence from a reference policy. This process demands extensive sampling during training, significant hyperparameter tuning, and is prone to optimization instability.

**Method and Recipe**
Direct Preference Optimization (DPO) eliminates the explicit reward model and RL loop by deriving a closed-form mapping between reward functions and optimal policies under the KL-constrained objective. The algorithm optimizes the policy directly using a simple classification loss. The DPO training recipe proceeds as follows:
1. Initialize a reference policy $\pi_{\mathrm{ref}}$, typically a supervised fine-tuned (SFT) model. If unavailable, $\pi_{\mathrm{ref}}$ can be initialized by maximizing the likelihood of preferred completions in the dataset.
2. Construct an offline preference dataset $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$ containing prompts $x$ with human-preferred ($y_w$) and dispreferred ($y_l$) completions sampled from $\pi_{\mathrm{ref}}$.
3. Optimize the parameterized policy $\pi_\theta$ by minimizing the DPO loss, which treats preference learning as a binary cross-entropy objective over the policy's log-probability ratios relative to the reference model.
4. The gradient update implicitly increases the likelihood of preferred responses and decreases dispreferred ones, weighted dynamically by how incorrectly the implicit reward orders the completions, thereby preventing model collapse without explicit KL regularization during training.

**Key Formulas**
The foundation of DPO rests on the analytical solution to the KL-constrained reward maximization objective:
\[
\pi_{r}(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right),
\]
where $Z(x)$ is the partition function. By rearranging this to express the reward in terms of the optimal policy, $r(x, y) = \beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x)$, and substituting into the Bradley-Terry preference model, the partition function cancels. This yields the DPO objective:
\[
\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{w} \mid x)}{\pi_{\mathrm{ref}}(y_{w} \mid x)} - \beta \log \frac{\pi_{\theta}(y_{l} \mid x)}{\pi_{\mathrm{ref}}(y_{l} \mid x)} \right) \right],
\]
where $\sigma$ is the logistic function and $\beta$ controls the strength of the implicit KL constraint.

**Quantitative Results**
DPO matches or exceeds PPO-based RLHF across multiple benchmarks with minimal hyperparameter tuning. In controlled sentiment generation, DPO achieves a strictly superior reward-KL frontier, outperforming both standard PPO and PPO with ground-truth rewards. For TL;DR summarization using a GPT-J SFT model, DPO attains a 61% win rate at temperature 0.0, surpassing PPO's 57% at its optimal temperature, while demonstrating greater robustness to sampling temperature variations. In single-turn dialogue on the Anthropic HH dataset (Pythia-2.8B), DPO is the only computationally efficient method to improve over the dataset's preferred completions, matching the performance of a "Best of 128" baseline. Under out-of-distribution conditions on CNN/DailyMail articles, DPO achieves GPT-4 win rates of 0.36 (temp 0) and 0.31 (temp 0.25), significantly outperforming PPO (0.26 and 0.23). Human evaluations on summarization samples confirm DPO is preferred 58% of the time over PPO.

**Stated Limitations**
The authors note several limitations and directions for future work. The out-of-distribution generalization capabilities of DPO relative to explicit reward learning require more comprehensive study. It remains unclear how reward over-optimization manifests within the direct preference optimization framework. Experiments were limited to models up to 6B parameters, leaving scaling to larger architectures unexplored. Additionally, the authors acknowledge that automated GPT-4 evaluation win rates are sensitive to prompt phrasing, necessitating better methods for eliciting high-quality judgments. Finally, extending DPO to self-labeling with unlabeled prompts and applying it to non-text modalities are identified as promising research avenues.
