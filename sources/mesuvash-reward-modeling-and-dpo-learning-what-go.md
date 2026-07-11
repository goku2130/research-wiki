---
id: mesuvash:reward-modeling-and-dpo-learning-what-go
type: web
title: 'Reward Modeling and DPO: Learning What "Good" Means'
url: https://mesuvash.github.io/blog/2026/reward-modeling/
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-and-preference-optimization
---

The provided text, "Reward Modeling and DPO: Learning What 'Good' Means," by Mesuvash, details the methodologies for training language models to align with human preferences, primarily through Reward Modeling (RM) and Direct Preference Optimization (DPO).

### Core Problem

The core problem addressed is the difficulty of defining and optimizing for subjective alignment goals (e.g., helpfulness, safety, tone, factuality) in open-ended language model responses, where no simple programmatic function can provide a reward signal. Traditional Reinforcement Learning (RL) requires a reward function, but for many human-centric qualities, a ground truth score is unavailable.

### Method/Recipe Step by Step

#### 1. Reward Modeling (RM)

Reward Modeling learns a reward function from human preferences to provide a scalar score for a given prompt-response pair.

*   **Data Collection (Preference Data):**
    *   **Format:** Each data point consists of a prompt, a chosen (preferred) response, and a rejected (dispreferred) response.
    *   **Generation:** Responses (2-8 per prompt) are sampled from an initial Supervised Fine-Tuned (SFT) model or a mix of models.
    *   **Annotation:** Human annotators compare responses to a prompt and select the better one. Guidelines define "better." Rankings of $k > 2$ responses are decomposed into pairwise comparisons.
    *   **Quality Control:** Includes inter-annotator agreement checks (20-35% disagreement is normal), majority voting, and filtering.
    *   **Dataset Sizes:** Production models use 100k-500k pairs (e.g., InstructGPT ~50k, Anthropic ~170k, Llama 2 >1M).

*   **Reward Model Architecture:**
    *   A pre-trained Large Language Model (LLM) (often an SFT checkpoint) is used.
    *   The final vocabulary projection layer is replaced with a linear layer that maps the last hidden state (at the EOS token) to a single scalar output, $R(x,y)$.
    *   Input is the concatenation of prompt $x$ and response $y$.

*   **Training Recipe:**
    *   **Initialization:** From an SFT checkpoint or pre-trained base model.
    *   **Head Replacement:** Remove the LM head (shape $d \times V$) and add a randomly initialized linear layer (shape $d \times 1$).
    *   **Fine-tuning:** Standard supervised training on preference pairs. Learning rate: $1 \times 10^{-5}$ to $5 \times 10^{-6}$. Train for 1-2 epochs to avoid overfitting. Batch sizes: 64-128 pairs.
    *   **Evaluation:** Report accuracy on a held-out set (10-20% of data); good RMs achieve 70-75% accuracy (human agreement is 73-80%).

*   **Reward Normalization:**
    *   Per-batch normalization (subtract mean, divide by std dev).
    *   Clipping rewards (e.g., to $[-5, 5]$).
    *   Length penalty or normalization to counter length bias.

#### 2. Direct Preference Optimization (DPO)

DPO directly optimizes the policy using preference data, bypassing the need for a separate reward model and RL training loop.

*   **Pipeline:** Preference data $\rightarrow$ Supervised training on pairs $\rightarrow$ Aligned model.
*   **Training:** For each preference pair $(x, y_w, y_l)$, DPO computes log-probabilities of both responses under the current policy $\pi_\theta$ and a frozen reference policy $\pi_{ref}$. The gradient simultaneously increases the probability of $y_w$ and decreases the probability of $y_l$ (relative to $\pi_{ref}$).

### Key Formulas in LaTeX

#### Reward Model Training Loss (Bradley-Terry Model):

The probability that $y_w$ is preferred over $y_l$ given prompt $x$ is:

$$
P(y_w \succ y_l \mid x) = \sigma\big(R_\phi(x, y_w) - R_\phi(x, y_l)\big)
$$

The negative log-likelihood loss for training the reward model $R_\phi$:

$$
\mathcal{L}_\text{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\big[\log \sigma\big(R_\phi(x, y_w) - R_\phi(x, y_l)\big)\big]
$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the sigmoid function.

#### KL-Penalized Reward (for RLHF):

The total reward used in RLHF, incorporating a KL divergence penalty to prevent policy drift:

$$
R_\text{total}(x,y) = R_\phi(x,y) - \beta \cdot D_\text{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_\text{ref}(\cdot \mid x)\big)
$$

where $\pi_\theta$ is the current policy, $\pi_\text{ref}$ is the reference policy (e.g., SFT model), and $\beta$ is the KL penalty coefficient.

#### DPO Loss:

The DPO loss directly optimizes the policy $\pi_\theta$ using preference pairs:

$$
\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right]
$$

Here, $\log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}$ acts as an implicit reward.

#### IPO Loss:

IPO replaces DPO's sigmoid loss with a squared loss to prevent overfitting:

$$
\mathcal{L}_\text{IPO}(\theta) = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} - \frac{1}{2\beta}\right)^2\right]
$$

#### KTO Loss (Simplified):

KTO works with unpaired "good" or "bad" labels:

$$
\mathcal{L}_\text{KTO}(\theta) = \mathbb{E}_{y_w}\left[\sigma\left(-r_\theta(x, y_w)\right)\right] + \mathbb{E}_{y_l}\left[\sigma\left(r_\theta(x, y_l)\right)\right]
$$

where $r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)} - \mathbb{E}_{y'}\left[\beta \log \frac{\pi_\theta(y' \mid x)}{\pi_\text{ref}(y' \mid x)}\right]$.

### Key Quantitative Results and Numbers

*   **Reward Model Accuracy:** Good RMs achieve 70-75% accuracy on held-out preference data. Human agreement on these datasets is typically 73-80%.
*   **KL Penalty $\beta$:** Typical values range from 0.01 to 0.2.
*   **Learning Rates for RM:** $1 \times 10^{-5}$ to $5 \times 10^{-6}$.
*   **Batch Sizes for RM:** 64-128 pairs.
*   **Training Epochs for RM:** 1-2 epochs to avoid overfitting.
*   **Reward Model Sizing:** Typically same size or slightly smaller than the policy model (e.g., InstructGPT: 6B RM for 175B policy; Llama 2: 70B RM for 70B policy; Anthropic: up to 52B).
*   **Annotator Disagreement:** 20-35% disagreement is normal for subjective tasks.
*   **Compute Cost:** RLHF is 3-5x the cost of SFT. DPO is 1.5-2x the cost of SFT.

### Stated Limitations

#### Reward Modeling Limitations:

*   **Preference Data Ceiling:** If annotators cannot reliably distinguish good from great responses, neither can the RM. This is particularly problematic for technical topics where annotator expertise may be limited.
*   **Reward Hacking:** The policy can exploit imperfections in the reward model, leading to outputs that score high but are undesirable to humans (e.g., verbosity, sycophancy, formatting tricks, hedging).
*   **Position Bias:** Annotators may prefer the first response shown.
*   **Length Bias:** Annotators may prefer longer responses.
*   **Annotator Disagreement:** Training on majority vote or random choices smooths over genuine preference diversity.

#### DPO Limitations:

*   **Offline Algorithm:** DPO trains on a fixed dataset and cannot explore or generate new responses. It cannot correct new types of errors not represented in the training data.
*   **Bounded by Dataset Quality:** Its ceiling is limited by the quality and diversity of the preference dataset.
*   **Overfitting (DPO specifically):** The sigmoid loss can saturate, leading to overfitting to preference pairs and vanishing gradients once a pair is "right." IPO attempts to mitigate this.
