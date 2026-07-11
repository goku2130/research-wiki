---
id: rlhfbook:rlhf-book-reward-models-section
type: web
title: 'RLHF Book: Reward Models Section'
url: https://rlhfbook.com/c/05-reward-models
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-modeling
---

# Reward Modeling in RLHF

### Core Problem
In Reinforcement Learning from Human Feedback (RLHF), the core challenge is learning from complex, hard-to-specify human preferences. Reward models (RMs) address this by acting as proxy objectives for optimization. They compress high-dimensional human preference data into a scalar representation, effectively serving as a learned environment that provides the learning signal for the agent, replacing the fixed reward functions found in traditional reinforcement learning.

### The Bradley-Terry Reward Model
The most common implementation is the Bradley-Terry (BT) model, which predicts the probability that a human judge prefers one completion over another.

#### Method and Recipe
1.  **Architecture**: A language model is converted into a scalar scorer by appending a small linear head to the model's final hidden state (typically the end-of-sequence/EOS token).
2.  **Input**: The model takes a prompt $x$ and two sampled completions, $y_c$ (chosen/preferred) and $y_r$ (rejected).
3.  **Scoring**: The reward model $r_\theta$ assigns a scalar score to each completion: $r_\theta(y_c|x)$ and $r_\theta(y_r|x)$.
4.  **Optimization**: The model is trained via maximum likelihood estimation to maximize the probability that the chosen completion has a higher score than the rejected one.
5.  **Training Schedule**: To prevent overfitting, the standard practice is to train the reward model for only one epoch.

#### Key Formulas
The probability that item $i$ is preferred over item $j$ is defined as:

$$
P(i > j) = \frac{p_i}{p_i + p_j}
$$

By reparameterizing latent strength as $p_i = e^{r_i}$, the preference probability depends on the score difference via the sigmoid function $\sigma$:

$$
P(i > j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)
$$

The resulting loss function is the negative log-likelihood of the observed preference:

$$
L(\theta) = -\log(\sigma(r_\theta(y_c|x) - r_\theta(y_r|x)))
$$

An equivalent form using the softplus function is:

$$
L(\theta) = \log(1 + e^{r_\theta(y_r|x) - r_\theta(y_c|x)})
$$

### Reward Model Variants
*   **Preference Margin Loss**: Used in Llama 2 to incorporate the magnitude of preference (e.g., from Likert scales) by introducing a margin $m(y_c, y_r)$:

$$
L(\theta) = -\log(\sigma(r_\theta(y_c|x) - r_\theta(y_r|x) - m(y_c, y_r)))
$$

*   **Balanced Multiple Comparisons**: To prevent overfitting when $K$ completions are sampled per prompt (creating $\binom{K}{2}$ pairs), InstructGPT averages the loss across all pairs for a single prompt so that each prompt contributes one grouped update.
*   **K-Wise Loss**: Based on the Plackett-Luce model (used in Starling 7B/34B), this generalizes pairwise comparisons to a complete ranking of $K$ items.

### Outcome Reward Models (ORM)
For reasoning-heavy tasks, ORMs are used to predict whether a completion is correct (1) or incorrect (0).

#### Method and Recipe
1.  **Architecture**: Similar to a standard RM, but the scalar head outputs predictions on a **per-token basis** rather than a single sequence-level score.
2.  **Labeling**: Correct completions are labeled 1 and incorrect ones 0. These labels are copied onto every completion token, while prompt tokens are masked (typically with -100).
3.  **Optimization**: The model is trained using a per-token binary cross-entropy loss.

#### Key Formula

$$
L_{CE}(\theta) = -\mathbb{E}_{(s,r) \sim D} [r \log p_\theta(s) + (1-r) \log(1 - p_\theta(s))]
$$

where $r \in \{0,1\}$ is the binary correctness label and $p_\theta(s)$ is the predicted probability of correctness.

### Limitations
*   **Overfitting**: Reward models are highly susceptible to overfitting, necessitating short training durations (1 epoch).
*   **Diminishing Returns**: While margin loss was used in Llama 2, the Llama 3 team removed it after observing diminishing improvements upon scaling.
*   **ORM Noise**: Because ORMs judge based only on the final answer, the per-token updates can be noisy as they do not explicitly capture intermediate reasoning errors.
