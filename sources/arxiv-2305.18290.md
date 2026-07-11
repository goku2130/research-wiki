---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-and-preference-optimization
---

# Direct Preference Optimization (DPO)

### Core Problem
The standard pipeline for aligning Large Language Models (LMs) with human preferences—Reinforcement Learning from Human Feedback (RLHF)—is computationally expensive and often unstable. RLHF typically requires a three-stage process: supervised fine-tuning (SFT), training a separate reward model to reflect human preferences, and optimizing the LM using reinforcement learning (e.g., Proximal Policy Optimization, or PPO) to maximize that reward while maintaining a KL-divergence constraint to prevent the model from drifting too far from the reference policy. This process involves training multiple models, sampling from the LM during fine-tuning, and extensive hyperparameter tuning.

### Method
Direct Preference Optimization (DPO) eliminates the need for an explicit reward model and the RL training loop. The authors derive an analytical mapping between reward functions and optimal policies, allowing the preference optimization problem to be solved as a simple classification task.

**The DPO Recipe:**
1. **Data Collection:** Construct an offline dataset of preferences $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$, where $x$ is a prompt, $y_w$ is the preferred completion, and $y_l$ is the dispreferred completion.
2. **Reference Model Initialization:** Initialize a reference policy $\pi_{\text{ref}}$. This is typically the SFT model. If an SFT model is unavailable, $\pi_{\text{ref}}$ is initialized by maximizing the likelihood of preferred completions.
3. **Policy Optimization:** Optimize the language model $\pi_\theta$ by minimizing the DPO loss function using a binary cross-entropy objective. This implicitly fits a reward model while directly updating the policy.

### Key Formulas
DPO leverages the Bradley-Terry model for preferences, where the probability that $y_1$ is preferred over $y_2$ is:

$$
p^*(y_1 \succ y_2 \mid x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}
$$

The authors show that the optimal policy $\pi_r$ for the KL-constrained reward maximization problem takes the form:

$$
\pi_{r}(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)
$$

By rearranging this to express the reward $r$ in terms of the policy, the DPO objective is formulated as:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{w} \mid x)}{\pi_{\mathrm{ref}}(y_{w} \mid x)} - \beta \log \frac{\pi_{\theta}(y_{l} \mid x)}{\pi_{\mathrm{ref}}(y_{l} \mid x)} \right) \right]
$$

where $\beta$ is a parameter controlling the strength of the KL penalty and $\sigma$ is the logistic function.

### Key Quantitative Results
DPO was evaluated on models up to 6B parameters across three tasks:

*   **Controlled Sentiment Generation:** DPO's reward-KL frontier strictly dominated PPO, achieving higher rewards for the same KL divergence. DPO outperformed PPO even when PPO had access to ground-truth rewards (PPO-GT).
*   **Summarization (TL;DR):** DPO achieved a win rate of approximately $61\%$ (at temperature 0.0) against reference completions, exceeding PPO's $57\%$. DPO was also found to be more robust to changes in sampling temperature.
*   **Single-Turn Dialogue (Anthropic HH):** DPO was the only computationally efficient method to improve over the preferred completions in the test set, matching or exceeding the performance of the computationally demanding "Best of 128" baseline.
*   **Out-of-Distribution Generalization:** When tested on CNN/DailyMail news articles (having been trained on Reddit TL;DR), DPO maintained a higher win rate against ground-truth summaries than PPO (e.g., $0.36$ vs $0.26$ at temperature 0).

### Stated Limitations
*   **OOD Generalization:** While initial results are promising, the authors state that more comprehensive study is needed to understand how DPO generalizes compared to explicit reward functions.
*   **Self-Labeling:** It remains an open question whether DPO can effectively utilize unlabeled prompts via self-labeling, as is done in some RLHF frameworks.
*   **Reward Over-optimization:** The authors note a slight decrease in performance during extended training (Figure 3-right), which may be an instance of reward over-optimization.
*   **Scalability:** The experiments were limited to models up to 6B parameters; scaling to significantly larger state-of-the-art models is suggested for future work.
*   **Evaluation Sensitivity:** The win rates computed by GPT-4 were found to be impacted by the specific prompt used for evaluation.
