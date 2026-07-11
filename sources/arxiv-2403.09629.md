---
id: arxiv:2403.09629
type: paper
title: 'Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking'
url: https://arxiv.org/abs/2403.09629
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

### Core Problem
Traditional methods for teaching language models (LMs) to reason, such as the Self-Taught Reasoner (STaR), typically rely on curated question-answering (QA) datasets. This approach limits the scale and generalizability of the learned rationales to specific tasks. The authors of Quiet-STaR argue that reasoning is implicit in almost all written text and propose a method for LMs to learn to generate internal rationales from diverse, unstructured internet text to improve their predictions of future tokens.

### Method
Quiet-STaR generalizes the STaR framework by training an LM to "think" (generate rationales) at every token position to better predict subsequent text. The process follows a three-step cycle:

1.  **Parallel Rationale Generation (Think):** To avoid the computational cost of sequential forward passes, the model uses a parallel sampling algorithm. For each token $x_i$ in a sequence, the model generates $r$ rationale candidates of length $t$. These are demarcated by learned `<|startofthought|>` and `<|endofthought|>` tokens. A diagonal attention mask is used to ensure that thought tokens only attend to their own rationale path and the preceding text.
2.  **Mixing Predictions (Talk):** Because initial thoughts are often out-of-distribution and can harm performance, a "mixing head" (a shallow MLP) is introduced. This head outputs a scalar weight that interpolates between the base LM's next-token prediction and the prediction made after the rationale.
3.  **Optimizing Rationale Generation (Learn):** The model is optimized using the REINFORCE algorithm. Rationales are rewarded if they increase the log-likelihood of future tokens compared to the average rationale for that position. To reduce variance and provide a stronger signal, the authors employ a non-myopic loss via teacher-forcing, supervising the prediction of multiple tokens ahead ($n_{true}$) rather than just the immediate next token.

### Key Formulas
The general objective is to find parameters $\theta$ that maximize the likelihood of the remaining sequence given a generated rationale:

$$
\theta^* = \arg \max_\theta \mathbb{E}[\log p_\theta(x_{i:n} | x_{0:i}, \text{rationale}_\theta(x_{0:i}))]
$$

The reward $r_j$ for a specific rationale $T_j$ is defined as the difference between the log-likelihood of the future tokens under that rationale and the average log-likelihood across all sampled rationales for that token:

$$
r_{j}=\log p_{j:j+n_{true}}^{\text{talk}}(X_{j+1:j+n_{true}+1})-\log\overline{p}_{j:j+n_{true}}^{\text{talk}}(X_{j+1:j+n_{true}+1})
$$

The REINFORCE loss used to update the LM parameters and the special thought token embeddings is:

$$
\nabla_{\theta}\mathcal{L}_{j}^{\text{REINFORCE}}=-r_{j}\cdot\nabla_{\theta}\log p_{\theta}(T_{j}|[X_{:j}; <\text{startofthought}>])
$$

### Key Quantitative Results
The authors applied Quiet-STaR to Mistral 7B using the OpenWebMath and C4 datasets. They observed that zero-shot reasoning capabilities improved without any task-specific fine-tuning:

*   **OpenWebMath Training:**
    *   **GSM8K:** Accuracy increased from $5.9\% \rightarrow 10.9\%$.
    *   **CommonsenseQA:** Accuracy increased from $36.3\% \rightarrow 47.2\%$.
*   **C4 Training:**
    *   **GSM8K:** Accuracy increased from $5.9\% \rightarrow 8.1\%$.
    *   **CommonsenseQA:** Accuracy increased from $36.3\% \rightarrow 42.6\%$.
*   **Chain-of-Thought (CoT) Integration:** When using internal rationales to augment zero-shot CoT on GSM8K, the majority vote accuracy over 8 samples (cot-maj@8) increased from $40.6\%$ to $47.7\%$.

The authors noted that improvements consistently scaled with the number of thinking tokens used during training and that the model disproportionately improved its prediction of "difficult" tokens (e.g., recalling theorems or proof steps).

### Stated Limitations
*   **Computational Overhead:** The process of generating multiple rationales before every output token introduces substantial compute costs.
*   **Model Scale:** The method was only tested on a 7B parameter model; the authors suggest larger models might yield disproportionately better results.
*   **Initialization:** The study used a pre-trained model; it remains unknown if these techniques work when training a model from scratch.
*   **Static Thought Generation:** The current implementation does not dynamically predict when to start or end a rationale; it uses a fixed length and frequency.
