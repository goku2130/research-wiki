---
id: arxiv:2505.13697
type: paper
title: RL in Name Only? Analyzing the Structural Assumptions in RL post-training for
  LLMs
url: https://arxiv.org/html/2505.13697v4
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

The authors of "RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs" investigate the foundational assumptions of applying Reinforcement Learning (RL) to Large Language Model (LLM) post-training, specifically focusing on the Group Relative Policy Optimization (GRPO) algorithm. They argue that common structural assumptions in the LLM-MDP (Markov Decision Process) formulation lead to a degenerate MDP, effectively reducing the problem to a contextual bandit and GRPO to a variant of Filtered Iterative Supervised Fine-Tuning (F-ISFT).

**Core Problem:**
The paper identifies two critical structural assumptions in the popular LLM-MDP formulation that undermine the true RL nature of post-training:
1.  **States as sequences of actions:** The MDP state is defined as a concatenation of previously generated tokens, meaning each state explicitly encodes the entire history of actions. This deterministic transition and full history in the state space transforms the sequential decision-making into a process of constructing a single "macro-action" (the complete response).
2.  **Terminal reward and uniform credit assignment:** Rewards are binary (1 for correct, 0 for incorrect) and assigned only at the terminal state by an external verifier. Crucially, this terminal reward (or derived advantage) is then uniformly distributed across all tokens in the generated sequence. This short-circuits the temporal credit assignment problem inherent in traditional RL.

These assumptions, the authors contend, cause the LLM-MDP to behave like a contextual bandit problem, where the input prompt is the context, and generating a full sequence is like pulling a single arm, with feedback received only at the end.

**Method/Recipe Step by Step (GRPO as F-ISFT):**
The authors theoretically demonstrate that under these structural assumptions, the GRPO objective simplifies to an on-policy variant of F-ISFT.

1.  **Start with GRPO Objective:** The original GRPO objective function is given by:

$$
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}, 1 - \varepsilon, 1 + \varepsilon \right) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right\} \right]
$$

    where $\mathcal{ISR}_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})}$ is the importance sampling ratio, and $\hat{A}_{i,t}$ is the standardized advantage.

2.  **Relax KL Penalty:** Given that the clipping operation already constrains policy updates, the KL penalty term ($\beta \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$) is often found to have limited effect and can be removed for simplification.

3.  **Assume ISR within Clipping Range:** If the importance sampling ratio $\mathcal{ISR}_{i,t}(\theta)$ is within the clipping range $(1 - \varepsilon, 1 + \varepsilon)$, the objective simplifies to:

$$
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \mathcal{ISR}_{i,t}(\theta) \hat{A}_{i,t} \right]
$$

4.  **Uniform Advantage Distribution:** Since the relative advantage $\hat{A}_{i,t}$ is uniformly distributed across all tokens for a given response $o_i$ (i.e., $\hat{A}_{i,t} = \hat{A}_i$), it can be pulled out of the inner summation:

$$
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{\hat{A}_i}{|o_i|} \sum_{t=1}^{|o_i|} \mathcal{ISR}_{i,t}(\theta) \right]
$$

5.  **Decomposition by Positive and Negative Responses:** For a given question $q$, responses are divided into positive ($\mathcal{G}^+$) and negative ($\mathcal{G}^-$) based on binary reward. The relative advantage scores for all positive responses are $\hat{A}_q^+$ and for all negative responses are $\hat{A}_q^-$. The objective can then be split:

$$
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac {1}{G} \left[ \sum_ {i = 1} ^ {|\mathcal {G} ^ {+}|} \frac {\hat {A} _ {q} ^ {+}}{| o _ {i} |} \sum_ {t = 1} ^ {| o _ {i} |} \mathcal {I S R} _ {i, t} (\theta) + \sum_ {i = 1} ^ {|\mathcal {G} ^ {-}|} \frac {\hat {A} _ {q} ^ {-}}{| o _ {i} |} \sum_ {t = 1} ^ {| o _ {i} |} \mathcal {I S R} _ {i, t} (\theta) \right] \right]
$$

    Defining $A_{q,i}^{+} = \frac{\hat{A}_{q}^{+}}{|o_{i}|}$ and $A_{q,i}^{-} = \frac{\hat{A}_{q}^{-}}{|o_{i}|}$, the objective becomes:

$$
\mathcal{J}(\theta) = \mathbb{E} \left[ \frac {1}{G} \left[ \sum_ {i = 1} ^ {|\mathcal {G} ^ {+}|} A _ {q, i} ^ {+} \sum_ {t = 1} ^ {H} \mathcal {I S R} _ {i, t} (\theta) + \sum_ {i = 1} ^ {|\mathcal {G} ^ {-}|} A _ {q, i} ^ {-} \sum_ {t = 1} ^ {H} \mathcal {I S R} _ {i, t} (\theta) \right] \right]
$$

    The gradient of this objective, using $\nabla_{\theta}\pi (\theta) = \pi (\theta)\nabla_{\theta}\log \pi (\theta)$, is:

$$
\nabla_ {\theta} \mathcal {J} (\theta) = \mathbb {E} \left[ \frac {1}{G} \left[ A ^ {+} \sum_ {i = 1} ^ {|\mathcal {G} ^ {+}|} \sum_ {t = 1} ^ {H} \mathcal {I S R} _ {i, t} (\theta) \nabla_ {\theta} \log \left(\pi_ {\theta} \left(o _ {i, t} \mid q, o _ {i, <   t}\right)\right) \right. \right.
$$

$$
\left. \left. + A ^ {-} \sum_ {i = 1} ^ {|\mathcal {G} ^ {-}|} \sum_ {t = 1} ^ {H} \mathcal {I S R} _ {i, t} (\theta) \nabla_ {\theta} l o g (\pi_ {\theta} (o _ {i, t} | q, o _ {i, <   t})) \right] \right]
$$

    This gradient update resembles weighted supervised learning, where positive responses increase log-likelihood and negative responses decrease it, with weights $A^+$ and $A^-$.

**Key Quantitative Results and Numbers:**
The paper conducts experiments on GSM8K and Countdown datasets using Qwen-2.5 (0.5B, 1.5B), Llama-3.2-Instruct (1B, 3B), Deepseek-math-7B-Instruct, and Qwen3-8B models.

*   **Performance Equivalence:**
    *   On GSM8K (Qwen-2.5-0.5B): GRPO (55.87%), GRPO-wo-KL (55.19%), Filtered-ISFT+ (51.71%), Filtered-ISFT+- (55.72%).
    *   On GSM8K (Qwen-2.5-1.5B): GRPO (78.24%), GRPO-wo-KL (76.87%), Filtered-ISFT+ (74.98%), Filtered-ISFT+- (76.37%).
    *   On Countdown (Qwen-2.5-0.5B): GRPO (37.73%), GRPO-wo-KL (42.43%), Filtered-ISFT+ (44.01%), Filtered-ISFT+- (37.86%).
    *   On Countdown (Qwen-2.5-1.5B): GRPO (71.43%), GRPO-wo-KL (70.82%), Filtered-ISFT+ (53.57%), Filtered-ISFT+- (65.89%).
    *   On GSM8K (Deepseek-math-7B-Instruct): GRPO (82.4%), PPO (81.9%), Filtered-ISFT+- (83.7%).
    *   On GSM8K (Qwen3-8B): GRPO (92.1%), PPO (91.8%), Filtered-ISFT+- (91.5%).
    *   On GSM8K (Llama-3.2-1B-Instruct): GRPO (62.01%), Filtered-ISFT+ (57.31%), Filtered-ISFT+- (63.07%).
    *   On GSM8K (Llama-3.2-3B-Instruct): GRPO (84.59%), Filtered-ISFT+ (81.95%), Filtered-ISFT+- (83.54%).
    *   Filtered-ISFT- (trained only on negative samples) showed comparable performance on GSM8K but significantly underperformed on Countdown (0.00% for both Qwen-2.5 models), suggesting its inadequacy for more challenging tasks.

*   **Length Bias:**
    *   For correct responses, GRPO and Filtered-ISFT+- show a slight increase in average response length, while Filtered-ISFT+ remains relatively constant.
    *   For incorrect responses, the average length decreases initially but then increases substantially for GRPO and Filtered-ISFT+-, whereas Filtered-ISFT+ maintains comparable lengths.
    *   This indicates that training on incorrect responses (as in GRPO and F-ISFT+-) contributes to increased output length.

**Stated Limitations:**
*   The analysis focuses on the specific LLM-MDP formulation commonly used, particularly in DeepSeek-R1, and its implications.
*   The theoretical simplification of the GRPO objective assumes the KL penalty term has limited effect and that the importance sampling ratio is within the clipping range. While empirically supported, these are assumptions.
*   The paper highlights that the observed increase in response length in RL-trained LLMs is a side effect of the uniform advantage distribution and length scaling, rather than improved reasoning. This implies that current RL methods may not be imparting genuinely novel reasoning capabilities but rather refining existing ones.
*   The effectiveness of random rewards (as shown in other works) is clarified by showing they primarily affect advantage estimates, which are not essential for performance in the F-ISFT framework.
*   The analysis suggests that research directions attempting to fix length issues (e.g., length penalties) are addressing symptoms rather than the root cause inherent in the basic MDP formulation.
