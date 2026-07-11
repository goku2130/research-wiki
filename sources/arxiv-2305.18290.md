---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

# Direct Preference Optimization (DPO)

### Core Problem
The standard pipeline for Reinforcement Learning from Human Feedback (RLHF) is computationally expensive and often unstable. It typically requires three distinct phases: supervised fine-tuning (SFT), training a separate reward model to reflect human preferences, and optimizing the language model (LM) using reinforcement learning (RL) algorithms like Proximal Policy Optimization (PPO). This process involves sampling from the LM during training and requires significant hyperparameter tuning to prevent the model from drifting too far from the reference policy (mode collapse) or over-optimizing the reward.

### Method
Direct Preference Optimization (DPO) eliminates the need for an explicit reward model and RL training loop. The authors leverage a theoretical mapping between reward functions and optimal policies to express the RLHF objective directly in terms of the policy. By treating the LM as an implicit reward model, DPO solves the preference alignment problem using a simple binary cross-entropy classification loss.

**Step-by-Step Recipe:**
1. **Data Collection:** Construct an offline dataset of preferences $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$, where $x$ is a prompt, $y_w$ is the preferred completion, and $y_l$ is the dispreferred completion.
2. **Reference Model Initialization:** Initialize a reference policy $\pi_{\text{ref}}$. This is typically the SFT model. If an SFT model is unavailable, $\pi_{\text{ref}}$ is initialized by maximizing the likelihood of the preferred completions $(x, y_w)$.
3. **Policy Optimization:** Optimize the policy $\pi_\theta$ by minimizing the DPO loss function. This update increases the relative log probability of preferred responses over dispreferred ones, weighted by the current implicit reward difference to prevent model degeneration.

### Key Formulas
The optimal policy $\pi_r$ for a KL-constrained reward maximization objective is given by:

$$
\pi_{r}(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)
$$

where $Z(x)$ is the partition function and $\beta$ controls the deviation from the reference policy. DPO rearranges this to express the reward $r$ as:

$$
r(x, y) = \beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x)
$$

Substituting this into the Bradley-Terry preference model allows the partition function to cancel out, resulting in the **DPO Loss**:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{w} \mid x)}{\pi_{\mathrm{ref}}(y_{w} \mid x)} - \beta \log \frac{\pi_{\theta}(y_{l} \mid x)}{\pi_{\mathrm{ref}}(y_{l} \mid x)} \right) \right]
$$

### Key Quantitative Results
DPO was evaluated on models up to 6B parameters across three tasks:

*   **Controlled Sentiment Generation:** DPO produced a reward-KL frontier that strictly dominated PPO, achieving higher rewards for the same KL divergence, even when PPO had access to ground-truth rewards (PPO-GT).
*   **Summarization (TL;DR):** Using GPT-4 as an evaluator, DPO achieved a win rate of approximately $61\%$ at temperature $0.0$, exceeding PPO's $57\%$. DPO also demonstrated greater robustness to sampling temperature than PPO.
*   **Single-Turn Dialogue (Anthropic HH):** DPO was the only computationally efficient method that improved over the preferred completions in the test set, matching or exceeding the performance of the "Best of 128" baseline.
*   **Out-of-Distribution Generalization:** When tested on CNN/DailyMail news articles (having been trained on Reddit TL;DR), DPO maintained a higher win rate against ground-truth summaries than PPO ($0.36$ vs $0.26$ at temperature $0$).

### Stated Limitations
*   **OOD Generalization:** While initial results are promising, the authors state that more comprehensive study is needed to determine how DPO generalizes compared to explicit reward functions.
*   **Self-Labeling:** It is unknown if DPO can effectively utilize unlabeled prompts through self-labeling in the same way RLHF does.
*   **Over-optimization:** The authors note a slight decrease in performance late in training (Figure 3-right), which may be an instance of reward over-optimization.
*   **Scaling:** The method was only evaluated on models up to 6B parameters; scaling to state-of-the-art models is a future direction.
*   **Evaluation Bias:** The authors found that GPT-4 win rates are sensitive to the specific prompt used for evaluation.
