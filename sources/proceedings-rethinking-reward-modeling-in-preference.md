---
id: proceedings:rethinking-reward-modeling-in-preference
type: web
title: Rethinking Reward Modeling in Preference-based Large Language Models (ICLR
  2025)
url: https://proceedings.iclr.cc/paper_files/paper/2025/file/7423902b5534e2b267438c85444a54b1-Paper-Conference.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-modeling
---

# Rethinking Reward Modeling in Preference-based LLM Alignment

### Core Problem
The Bradley-Terry (BT) model is the standard for converting pairwise preference comparisons into reward values for Large Language Model (LLM) alignment. However, its application in LLM reward modeling differs significantly from its original use in multi-player stochastic games (like the LLM Chatbot Arena). In reward modeling, comparisons are extremely sparse—often with only $N/2$ comparisons for $N$ prompt-response pairs—and the model must predict rewards for unseen pairs. The authors investigate whether the BT model is theoretically sound in this sparse setting, whether it is necessary for downstream optimization, and whether the conventional practice of restricting comparisons to responses from the same prompt is optimal.

### Method and Recipe
The authors propose a shift from precise probability modeling to **order consistency**, arguing that a reward model only needs to preserve the correct ranking of responses through a monotonic transformation of the true reward.

**1. Theoretical Foundation**
The authors establish the first asymptotic theory for neural network-based BT regression. They model the reward function as $r(\Psi(x,y))$, where $\Psi$ is an embedding function. They utilize the **truncated KL risk** to provide a risk bound for reward estimation using Multi-Layer Perceptrons (MLPs).

**2. Classification-Based Reward Modeling**
As an alternative to the BT model's anti-symmetric structure, the authors propose a simple classification-based approach:
*   **Step 1:** Treat preference annotations as a binary classification task.
*   **Step 2:** Train a classifier (e.g., MLP or LightGBM) to predict the preference label $h$ for prompt-response pairs independently.
*   **Step 3:** Use the resulting logit of the classifier as a proxy for the reward value.
*   **Step 4:** Optimize the LLM using these reward proxies (evaluated via Best-of-N sampling).

**3. Annotation Strategy**
The authors compare two annotation designs:
*   **Same-prompt:** Comparing two responses generated from the same prompt.
*   **Cross-prompt:** Comparing two responses generated from different prompts.

### Key Formulas
The **Bradley-Terry (BT) model** defines the probability of selecting option $i$ over $j$ as:

$$
P(i \succ j) = \frac{\exp(r(i))}{\exp(r(i)) + \exp(r(j))} = \text{softmax}(r(i), r(j))
$$

To analyze convergence, the authors use the **$B$-truncated KL risk**:

$$
R_{B}(p_{0}, \hat{p}) = \mathbb{E} \left[ p_{0}^{\top} \min \left( B, \log \frac{p_{0}}{\hat{p}} \right) \right]
$$

The **Order Consistency** condition requires that for any two distinct pairs $(x_1, y_1)$ and $(x_2, y_2)$:

$$
(\hat{r}(x_1, y_1) - \hat{r}(x_2, y_2))(r(x_1, y_1) - r(x_2, y_2)) > 0
$$

Annotation quality is simulated using a sigmoid function $\xi$ based on the true reward difference $\Delta r$:

$$
\xi(\Delta r) = \sigma(\beta \Delta r)
$$

### Key Quantitative Results
The authors conducted over 12,000 experimental runs across 6 base LLMs (Gemma 2b, Gemma 7b, LLaMA3-8b, and their SFT versions) and 2 datasets (Anthropic-Harmless and Anthropic-Helpful).

*   **Classification vs. BT:** Classification-based reward models (CLF-MLP and CLF-LGB) generally outperformed BT-MLP in improving golden reward values via Best-of-N (BoN) sampling ($N=500$).
*   **Cross-Prompt Advantage:** Cross-prompt comparisons significantly outperformed same-prompt comparisons. This gain was most pronounced in "Similar Comparison" settings where responses to a single prompt lacked diversity.
*   **Robustness:** Classification models demonstrated superior resilience to increasing annotation error rates (varying $\beta$ from 0.5 to 10.0, corresponding to error rates from 5% to 38%) compared to BT models.
*   **Efficiency:** The proposed embedding-based reward models are computationally efficient; reproducing the 12,000 experiments required approximately 6,000 CPU-core hours.

### Stated Limitations
The authors note that their synthetic experimental setups—specifically the "Similar Comparison" and "Diversified Comparison" scenarios used to isolate the source of gains from cross-prompt annotations—are not realizable in practice without an existing reward model. In real-world scenarios, randomly sampled pairs are the only realistic setup for annotations.
