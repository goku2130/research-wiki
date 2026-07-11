---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-and-preference-optimization
---

The paper "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" introduces Direct Preference Optimization (DPO), a novel method for fine-tuning large language models (LMs) to align with human preferences.

**Core Problem:**
Existing methods for aligning LMs with human preferences, primarily Reinforcement Learning from Human Feedback (RLHF), are complex, unstable, and computationally expensive. RLHF typically involves two stages: first, training a separate reward model to capture human preferences, and then fine-tuning the LM using reinforcement learning (e.g., PPO) to maximize this estimated reward while maintaining proximity to the original model. This process requires training multiple LMs and sampling from the LM policy during fine-tuning, incurring significant computational costs and implementation complexity.

**Method/Recipe Step by Step:**
DPO simplifies the preference learning pipeline by reparameterizing the reward model in RLHF such that its corresponding optimal policy can be extracted in closed form. This allows the standard RLHF problem to be solved with a simple classification loss, eliminating the need for explicit reward modeling or reinforcement learning.

1.  **Supervised Fine-Tuning (SFT) (Optional but common):** An initial pre-trained LM is fine-tuned on high-quality data for the downstream task to obtain a base model, $\pi^{\mathrm{SFT}}$.
2.  **Preference Data Collection:** A dataset of human preferences $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$ is collected. For each prompt $x^{(i)}$, humans indicate a preferred response $y_w^{(i)}$ and a dispreferred response $y_l^{(i)}$ from a pair of generated responses. These responses are typically sampled from $\pi^{\mathrm{SFT}}$.
3.  **Reference Policy Initialization:**
    *   If $\pi^{\mathrm{SFT}}$ is available and used for sampling preference data, $\pi_{\mathrm{ref}}$ is initialized to $\pi^{\mathrm{SFT}}$.
    *   If $\pi^{\mathrm{SFT}}$ is not available, $\pi_{\mathrm{ref}}$ is initialized by maximizing the likelihood of preferred completions: $\pi_{\mathrm{ref}} = \arg \max_\pi \mathbb{E}_{(x, y_w) \sim \mathcal{D}} [\log \pi(y_w \mid x)]$.
4.  **DPO Optimization:** The language model $\pi_\theta$ is directly optimized to minimize the DPO loss function $\mathcal{L}_{\mathrm{DPO}}$, which is a binary cross-entropy objective. This objective implicitly optimizes the same KL-constrained reward maximization objective as traditional RLHF.

**Key Formulas in LaTeX:**

*   **Bradley-Terry Preference Model (Traditional RLHF):**
    $p ^ {*} (y _ {1} \succ y _ {2} \mid x) = \frac {\exp \left(r ^ {*} (x , y _ {1})\right)}{\exp \left(r ^ {*} (x , y _ {1})\right) + \exp \left(r ^ {*} (x , y _ {2})\right)}$
*   **Negative Log-Likelihood Loss for Reward Model (Traditional RLHF):**
    $\mathcal {L} _ {R} (r _ {\phi}, \mathcal {D}) = - \mathbb {E} _ {(x, y _ {w}, y _ {l}) \sim \mathcal {D}} [ \log \sigma (r _ {\phi} (x, y _ {w}) - r _ {\phi} (x, y _ {l})) ]$
*   **KL-Constrained Reward Maximization Objective (Traditional RLHF and DPO's implicit objective):**
    $\max _ {\pi_ {\theta}} \mathbb {E} _ {x \sim \mathcal {D}, y \sim \pi_ {\theta} (y | x)} [ r _ {\phi} (x, y) ] - \beta \mathbb {D} _ {\mathrm{KL}} [ \pi_ {\theta} (y \mid x) | | \pi_ {\text {ref}} (y \mid x) ]$
*   **Optimal Policy in Closed Form (Key to DPO):**
    $\pi_{r}(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)$
    where $Z(x) = \sum_{y} \pi_{\mathrm{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)$
*   **Reward Function Reparameterization (DPO's implicit reward):**
    $r(x, y) = \beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x)$
*   **Human Preference Probability in terms of Optimal Policy (DPO's foundation):**
    $p^{*}(y_{1} \succ y_{2} \mid x) = \sigma \left( \beta \log \frac{\pi^{*}(y_{1} \mid x)}{\pi_{\mathrm{ref}}(y_{1} \mid x)} - \beta \log \frac{\pi^{*}(y_{2} \mid x)}{\pi_{\mathrm{ref}}(y_{2} \mid x)} \right)$
*   **DPO Loss Function:**
    $\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{w} \mid x)}{\pi_{\mathrm{ref}}(y_{w} \mid x)} - \beta \log \frac{\pi_{\theta}(y_{l} \mid x)}{\pi_{\mathrm{ref}}(y_{l} \mid x)} \right) \right]$
*   **Implicit Reward Function (defined by DPO):**
    $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$

**Key Quantitative Results and Numbers:**

*   **Sentiment Generation:** DPO achieved the most efficient reward-KL frontier, strictly dominating PPO. DPO produced higher expected reward for all KL values compared to PPO, even when PPO had access to ground truth rewards (PPO-GT).
*   **Summarization (Reddit TL;DR):**
    *   DPO achieved a win rate of approximately **61%** at a temperature of 0.0, exceeding PPO's performance of **57%** at its optimal sampling temperature of 0.0.
    *   DPO also achieved a higher maximum win rate compared to the Best of N baseline.
    *   DPO was more robust to sampling temperature changes than PPO.
    *   In human evaluations, DPO samples (temp 0.25) were preferred **58%** of the time over PPO samples (temp 0).
*   **Single-Turn Dialogue (Anthropic Helpful and Harmless):**
    *   DPO was the only computationally efficient method that improved over the preferred completions in the Anthropic HH dataset.
    *   DPO provided similar or better performance to the computationally demanding Best of 128 baseline.
*   **Generalization (CNN/DailyMail summarization):** DPO policies outperformed PPO policies by a significant margin on an out-of-distribution dataset (CNN/DailyMail news articles), with DPO achieving a win rate of **0.36** (Temp 0) vs. PPO's **0.26** (Temp 0).
*   **GPT-4 Evaluation Validation:** GPT-4 judgments correlated strongly with human judgments. For TL;DR summarization, human agreement with GPT-4 (C) was **67%** (DPO vs PPO-0) and **79%** (SFT vs PPO-0), which was similar to or higher than inter-human annotator agreement (**65%** for DPO vs PPO-0).

**Stated Limitations:**

*   The paper does not comprehensively study how DPO policies generalize out of distribution compared to learning from an explicit reward function, although initial evidence suggests similar generalization.
*   It is unclear how reward over-optimization manifests in the DPO setting, and whether observed slight performance decreases are instances of it.
*   Experiments were conducted with models up to 6B parameters; scaling DPO to state-of-the-art models orders of magnitude larger is an area for future work.
*   The win rates computed by GPT-4 are impacted by the prompt used, indicating a need for further study on eliciting high-quality judgments from automated systems.
*   The application of DPO beyond language models to other modalities is an open area for future research.
