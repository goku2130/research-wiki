---
id: mbrenndoerfer:kl-divergence-penalty-in-rlhf-theory-imp
type: web
title: 'KL Divergence Penalty in RLHF: Theory & Implementation'
url: https://mbrenndoerfer.com/writing/kl-divergence-penalty-rlhf-training
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

# KL Divergence Penalty in RLHF: Theory & Implementation

## Core Problem
The primary challenge addressed is **reward hacking** during Reinforcement Learning from Human Feedback (RLHF). Because reward models are neural networks trained on finite datasets, they are imperfect approximations of human judgment. Without constraints, an optimizing policy can perform an "adversarial search" to discover outputs that achieve high reward scores but are qualitatively poor or bizarre by human standards. 

Additionally, the reward model often fails to explicitly measure foundational capabilities of the pre-trained model, such as coherence, tone, and register. Without a mechanism to anchor the policy, these general capabilities can be lost during the alignment process.

## Method and Implementation
The solution is the implementation of a **KL divergence penalty**, which acts as a "safety tether" keeping the fine-tuned policy ($\pi_\theta$) close to a fixed reference model ($\pi_{\text{ref}}$), typically the Supervised Fine-Tuning (SFT) model.

### Step-by-Step Recipe
1.  **Establish a Reference:** Maintain a frozen copy of the pre-trained/SFT model to serve as the reference distribution.
2.  **Sample Generation:** Generate responses $y$ from the current policy $\pi_\theta$ given a prompt $x$.
3.  **Per-Token Computation:** For every token $y_t$ in the generated sequence, calculate the log probability of that token under both the policy model and the reference model.
4.  **Calculate Divergence:** Compute the difference between these log probabilities to determine the per-token KL contribution.
5.  **Aggregate Sequence Penalty:** Sum the per-token contributions across the entire sequence length $T$ to find the total KL divergence for that response.
6.  **Reward Shaping:** Subtract this KL penalty (scaled by a coefficient) from the reward model's score to create a constrained objective.
7.  **Adaptive Control:** Adjust the KL coefficient during training—increasing it (tightening the tether) if the policy drifts dangerously, or decreasing it (loosening the tether) if progress stalls.

## Key Formulas
The general definition of KL divergence for discrete distributions $P$ and $Q$ is:

$$
D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}[\log P(x) - \log Q(x)]
$$

For autoregressive language models, the probability of a sequence $y$ given prompt $x$ is factorized as:

$$
\pi(y|x) = \prod_{t=1}^{T} \pi(y_t | x, y_{<t})
$$

The sequence-level KL divergence between the policy $\pi_\theta$ and reference $\pi_{\text{ref}}$ is computed as the expectation over samples from the policy:

$$
D_{KL}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} \right]
$$

The practical per-token KL contribution is simplified to:

$$
\text{KL}_t = \log \pi_\theta(y_t | x, y_{<t}) - \log \pi_{\text{ref}}(y_t | x, y_{<t})
$$

## Key Quantitative Results and Properties
While the source does not provide specific experimental benchmarks, it highlights several critical technical properties:
*   **Computational Efficiency:** Due to the expectation-based formulation, the KL penalty is tractable for large vocabularies (50,000 to 100,000 tokens) because it only requires evaluating log probabilities for tokens actually generated, rather than summing over the entire vocabulary.
*   **Non-negativity:** $D_{KL}(P \| Q) \geq 0$, ensuring the penalty only reduces the objective.
*   **Asymmetry:** The penalty is more severe when the policy assigns high probability to tokens the reference model considers unlikely, encouraging conservative behavior.
*   **Decomposability:** The sequence-level divergence decomposes into a sum of token-level divergences, allowing for fine-grained analysis.

## Stated Limitations
The effectiveness of the KL penalty depends heavily on the choice of the KL coefficient:
*   **Too Large (Short Tether):** Severely limits exploration and prevents the model from making necessary improvements in alignment.
*   **Too Small (Long Tether):** Fails to prevent reward hacking, allowing the policy to drift into regions where the reward model is poorly calibrated.
