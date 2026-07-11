---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

The paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" introduces Direct Preference Optimization (DPO), a novel method for fine-tuning large language models (LMs) to align with human preferences.

**Core Problem:**
Existing methods for aligning LMs with human preferences, particularly Reinforcement Learning from Human Feedback (RLHF), are complex and often unstable. RLHF typically involves two main stages:
1.  **Reward Model (RM) Training:** Fitting a separate reward model to human preference data, which is often based on the Bradley-Terry model. This involves minimizing a negative log-likelihood loss:

$$
\mathcal {L} _ {R} (r _ {\phi}, \mathcal {D}) = - \mathbb {E} _ {(x, y _ {w}, y _ {l}) \sim \mathcal {D}} [ \log \sigma (r _ {\phi} (x, y _ {w}) - r _ {\phi} (x, y _ {l})) ] \quad (2)
$$

    where $r_{\phi}(x,y)$ is the parameterized reward model, $\mathcal{D}$ is the dataset of preferred ($y_w$) and dispreferred ($y_l$) completions for a given prompt ($x$), and $\sigma$ is the logistic function.
2.  **RL Fine-Tuning:** Fine-tuning the LM policy ($\pi_{\theta}$) using reinforcement learning (e.g., PPO) to maximize the learned reward while staying close to an initial reference policy ($\pi_{\text{ref}}$, typically an SFT model). The objective is:

$$
\max _ {\pi_ {\theta}} \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\theta} (y | x)} [ r _ {\phi} (x, y) ] - \beta \mathbb {D} _ {\mathrm{KL}} [ \pi_ {\theta} (y \mid x) | | \pi_ {\text {ref}} (y \mid x) ] \quad (3)
$$

    This RL phase is computationally intensive, requires sampling from the LM during training, and involves significant hyperparameter tuning.

**Method/Recipe Step by Step (Direct Preference Optimization - DPO):**
DPO simplifies the preference learning pipeline by directly optimizing the policy using a simple classification loss, eliminating the need for an explicit reward model and reinforcement learning. The key insight is to leverage a specific parameterization of the reward model that allows for the closed-form extraction of the optimal policy.

1.  **Derivation of Optimal Policy:** The optimal solution to the KL-constrained reward maximization objective (Eq. 3) can be expressed in closed form as:

$$
\pi_{r}(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right) \quad (4)
$$

    where $Z(x) = \sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)$ is the partition function.
2.  **Reparameterization of Reward Function:** Rearranging Eq. 4 allows expressing the reward function in terms of its corresponding optimal policy $\pi_r$ and the reference policy $\pi_{\mathrm{ref}}$:

$$
r(x, y) = \beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x) \quad (5)
$$

3.  **DPO Objective Derivation:** By substituting this reparameterization into the Bradley-Terry preference model (Eq. 1), the partition function $Z(x)$ cancels out. This allows expressing the human preference probability directly in terms of the optimal policy $\pi^*$ and reference policy $\pi_{\mathrm{ref}}$:

$$
p^{*}(y_{1} \succ y_{2} \mid x) = \frac{1}{1 + \exp \left( \beta \log \frac{\pi^{*}(y_{2} \mid x)}{\pi_{\mathrm{ref}}(y_{2} \mid x)} - \beta \log \frac{\pi^{*}(y_{1} \mid x)}{\pi_{\mathrm{ref}}(y_{1} \mid x)} \right)} \quad (6)
$$

4.  **DPO Loss Function:** A maximum likelihood objective for a parameterized policy $\pi_{\theta}$ can then be formulated as a simple binary cross-entropy loss:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{w} \mid x)}{\pi_{\mathrm{ref}}(y_{w} \mid x)} - \beta \log \frac{\pi_{\theta}(y_{l} \mid x)}{\pi_{\mathrm{ref}}(y_{l} \mid x)} \right) \right] \quad (7)
$$

    This loss directly optimizes the policy $\pi_{\theta}$ to satisfy preferences, implicitly defining a reward function.
5.  **DPO Pipeline:**
    *   **Data Collection:** Obtain a dataset of human preferences $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$, where $y_w$ is the preferred completion and $y_l$ is the dispreferred completion for prompt $x$. If a $\pi^{\text{SFT}}$ model is available, it is used as $\pi_{\text{ref}}$. Otherwise, $\pi_{\text{ref}}$ is initialized by maximizing the likelihood of preferred completions.
    *   **Optimization:** Optimize the language model $\pi_{\theta}$ by minimizing $\mathcal{L}_{\mathrm{DPO}}$ using standard gradient-based methods. The gradient of the DPO loss is:

$$
\nabla_\theta \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = - \beta \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)) \left[ \nabla_\theta \log \pi(y_w \mid x) - \nabla_\theta \log \pi(y_l \mid x) \right] \right]
$$

        where $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ is the implicitly defined reward. This gradient increases the likelihood of preferred completions and decreases the likelihood of dispreferred completions, weighted by how incorrectly the implicit reward model currently orders them.

**Key Quantitative Results and Numbers:**

*   **Sentiment Generation:** DPO produced the "most efficient frontier" of reward vs. KL-divergence, achieving the highest reward for all KL values. DPO's reward/KL tradeoff "strictly dominates PPO," even when PPO had access to ground truth rewards (PPO-GT).
*   **Summarization (Reddit TL;DR):**
    *   DPO achieved a win rate of approximately **61%** (at temperature 0.0) against reference summaries, exceeding PPO's **57%** (at its optimal sampling temperature of 0.0).
    *   DPO also achieved a higher maximum win rate compared to the "Best of N" baseline.
    *   DPO was "much more robust to the sampling temperature than PPO."
    *   In human evaluations, DPO samples (temp 0.25) were preferred **58%** of the time over PPO samples (temp 0).
*   **Single-Turn Dialogue (Anthropic Helpful and Harmless):**
    *   DPO was the "only computationally efficient method that improves over the preferred completions" in the dataset.
    *   DPO provided "similar or better performance" to the computationally demanding "Best of 128" baseline.
*   **Generalization (CNN/DailyMail):** When evaluating summarization policies trained on Reddit TL;DR on the CNN/DailyMail dataset (out-of-distribution), DPO (win rate **0.36** at temp 0, **0.31** at temp 0.25) continued to outperform PPO (win rate **0.26** at temp 0, **0.23** at temp 0.25) by a significant margin against ground-truth summaries.
*   **GPT-4 Evaluation Validation:** Human agreement with GPT-4 judgments (using the "concise" prompt) was typically similar to or higher than inter-human annotator agreement. For DPO vs. PPO-0, GPT-4 (C)-H agreement was **67%** vs. H-H agreement of **65%**.

**Stated Limitations:**

*   **Generalization out of distribution:** While initial results suggest DPO policies generalize similarly to PPO, more comprehensive study is needed, including whether DPO can effectively use unlabeled prompts through self-labeling.
*   **Reward over-optimization:** The paper questions how reward over-optimization manifests in DPO and whether a slight decrease in performance observed in one experiment (Figure 3-right) is an instance of it.
*   **Scaling:** Experiments were conducted with models up to 6B parameters; scaling DPO to state-of-the-art models orders of magnitude larger is an exciting future direction.
*   **Evaluation Sensitivity:** GPT-4 win rates are impacted by the prompt used, suggesting a need for further study on eliciting high-quality judgments from automated systems.
*   **Applications beyond LMs:** The paper suggests DPO could be applied to training generative models in other modalities.
