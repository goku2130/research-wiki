---
id: arxiv:2511.12573
type: paper
title: Mitigating Length Bias in RLHF through a Causal Lens
url: https://arxiv.org/abs/2511.12573
retrieved: '2026-07-12'
maturity: comprehensive
topic: length-and-format-bias
---

# Mitigating Length Bias in RLHF through a Causal Lens

### Core Problem
Reinforcement Learning from Human Feedback (RLHF) reward models frequently exhibit **length bias**, a systematic tendency to assign higher reward scores to longer responses regardless of their actual informativeness or relevance. This occurs because reward models leverage spurious correlations in human preference data, conflating verbosity with quality. Consequently, policy models optimized via these reward signals prioritize verbosity over clarity, leading to unnecessarily long and less effective outputs.

### Causal Framework and Method
The authors propose a causal framework based on **Pearl’s Causal Hierarchy (PCH)** to disentangle semantic content ($C$) from response length ($L$). While standard RLHF relies on associational data ($P(y|x)$), the authors argue that mitigating bias requires counterfactual reasoning—asking how a reward would change if the same content were expressed at a different length.

The mitigation pipeline consists of three stages:

**1. Counterfactual Data Augmentation**
The authors generate synthetic response pairs to isolate variables:
*   **Content-fixed augmentation:** Generates responses that preserve the original meaning while varying length. Techniques include filler insertion/deletion, pleonasm, redundant sentence pruning, and paraphrasing.
*   **Length-fixed augmentation:** Generates responses that maintain the same length bin but degrade semantic quality (e.g., removing necessary details or substituting information).
*   **Verification:** Length is verified via token-level binning (five quantiles: Very Short to Very Long). Semantic fidelity is verified using a binary classifier based on `all-mpnet-base-v2`.

**2. Bias Diagnosis**
The model is tested for **preference flips**. A flip occurs when the reward model reverses its ranking of a response pair solely due to a change in length while content remains constant. The **flip ratio** $F$ is defined as:

$$
F_{(A,B)} = \frac{\# \text{ of flipped preferences}}{\text{Total counterfactual comparisons}}
$$

A pair is diagnosed as length-biased if $F > 0.5$.

**3. Bias Mitigation (Retraining)**
The reward model is retrained using curated counterfactuals:
*   **Correcting Flips:** For biased pairs, the authors create a new pair using a content-fixed variant that matches the length of the previously preferred response, then revise the supervision label to favor the semantically superior response.
*   **Semantic Supervision:** Length-fixed variants are used to force the model to distinguish fine-grained semantic differences under identical stylistic conditions.

### Key Formulas
The reward model is trained using a Bradley–Terry style preference likelihood:

$$
P(T_1 \succ T_2 | X) = \sigma(R(X, T_1) - R(X, T_2))
$$

The training objective utilizes a margin-based ranking loss:

$$
L_{RM} = \max(0, m - R(X, T_{\text{chosen}}) + R(X, T_{\text{rejected}}))
$$

The goal of the causal intervention is to produce a reward function $R$ where:

$$
\frac{\partial R(X, T)}{\partial L} \approx 0 \quad \text{while} \quad \frac{\partial R(X, T)}{\partial C} \neq 0
$$

### Quantitative Results
*   **Bias Prevalence:** In a sample of 49,861 pairs, 23,651 (**47.43%**) exhibited length bias.
*   **Reward Model Performance:** On RewardBench-1, the `CDA_HRO` model achieved the highest mean accuracy. In length-controlled (LC) accuracy tests, `CDA_OpenLM` and `CDA_HRO` substantially outperformed the baseline `HRO`.
*   **Policy Model Impact:** When used to train a policy via PPO, `PPO_CDA_HRO` achieved a length-controlled winrate on AlpacaEval of **37.18%**, more than doubling the winrate of `PPO_HRO` (**18.97%**) and `SFT` (**16.97%**).
*   **Data Scale:** The final mitigation dataset consisted of 412,286 unique triplets, including 198,778 flipped content-fixed pairs and 213,699 aligned length-fixed augmentations.

### Limitations
The proposed method assumes a clean, realizable separation between semantic content and response length. The authors note that while this approach addresses verbosity, future extensions are required to handle other confounding factors such as tone, coherence, or factuality.
