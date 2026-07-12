---
id: arxiv:2403.09629
type: paper
title: 'Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking'
url: https://arxiv.org/abs/2403.09629
retrieved: '2026-07-12'
maturity: comprehensive
topic: rl-for-reasoning
---

# Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

Quiet-STaR is a generalization of the Self-Taught Reasoner (STaR) framework. While STaR focuses on bootstrapping reasoning from curated question-answering (QA) datasets, Quiet-STaR enables language models (LMs) to learn to generate internal rationales (thoughts) from arbitrary, unstructured text. The goal is to allow the model to "think" at every token position to improve its predictions of future text.

### Core Problem
Reasoning is often implicit in written text (e.g., the unstated steps in a proof). Traditional reasoning methods rely on task-specific datasets, which limits their scale and generalizability. Quiet-STaR addresses the challenge of teaching an LM to autonomously discover and utilize internal reasoning traces across diverse internet corpora without requiring curated reasoning tasks or human-annotated rationales.

### Method
Quiet-STaR implements a three-step cycle: **Think**, **Talk**, and **Learn**.

1.  **Parallel Rationale Generation (Think):**
    To avoid the computational cost of sequential forward passes, the model generates $r$ rationales of length $t$ in parallel across $n$ tokens in an input sequence. It uses learned $\langle\text{startofthought}\rangle$ and $\langle\text{endofthought}\rangle$ tokens to demarcate rationales. A diagonal attention mask is employed so that each generated thought attends only to the preceding text and its own internal tokens, preventing interference between "counterfactual" rationale paths.

2.  **Mixing Predictions (Talk):**
    Because initial thoughts are out-of-distribution and may degrade performance, a "mixing head" (a shallow MLP) is introduced. This head takes the hidden states of the original text token and the end-of-thought token to produce a scalar weight $w$. This weight interpolates between the base LM's next-token prediction and the prediction made after the rationale.

3.  **Optimizing Rationale Generation (Learn):**
    The model uses the REINFORCE algorithm to reward rationales that increase the likelihood of future tokens. To reduce variance and improve the signal, the authors use a "non-myopic" loss via teacher-forcing, supervising the prediction of multiple tokens ahead ($n_{true}$) rather than just the immediate next token.

### Key Formulas
The objective is to find parameters $\theta$ that maximize the likelihood of future tokens given an internal rationale:

$$
\theta^* = \arg \max_\theta \mathbb{E}[\log p_\theta(x_{i:n} | x_{0:i}, \text{rationale}_\theta(x_{0:i}))]
$$

The reward $r_j$ for a rationale $T_j$ is the difference between the log-likelihood of the future tokens under the mixed "talk" distribution and the average log-likelihood across all sampled rationales for that token:

$$
r_j = \log p_{j:j+n_{true}}^{\text{talk}}(X_{j+1:j+n_{true}+1}) - \log \overline{p}_{j:j+n_{true}}^{\text{talk}}(X_{j+1:j+n_{true}+1})
$$

The REINFORCE loss is then defined as:

$$
\nabla_\theta \mathcal{L}_j^{\text{REINFORCE}} = -r_j \cdot \mathbb{1}[r_j > 0] \cdot \nabla_\theta \log p_\theta(T_j | [X_{:j}; \langle\text{startofthought}\rangle])
$$

### Quantitative Results
The authors applied Quiet-STaR to Mistral 7B using the OpenWebMath and C4 datasets. Zero-shot improvements were observed without any task-specific fine-tuning:

*   **OpenWebMath Training:**
    *   **GSM8K:** Accuracy increased from $5.9\% \rightarrow 10.9\%$.
    *   **CommonsenseQA:** Accuracy increased from $36.3\% \rightarrow 47.2\%$.
*   **C4 Training:**
    *   **GSM8K:** Accuracy increased from $5.9\% \rightarrow 8.1\%$.
    *   **CommonsenseQA:** Accuracy increased from $36.3\% \rightarrow 42.6\%$.
*   **Chain-of-Thought (CoT) Integration:** When using internal rationales to augment zero-shot CoT on GSM8K, the majority vote accuracy (cot-maj@8) increased from $40.6\%$ to $47.7\%$.

The authors noted that performance consistently scaled with the number of thinking tokens used during training and that improvements were disproportionately concentrated on "difficult-to-predict" tokens.

### Limitations
*   **Computational Overhead:** Generating multiple internal tokens before every output token creates substantial overhead.
*   **Model Scale:** The method was only tested on a 7B parameter model; its efficacy on larger models remains to be seen.
*   **Training State:** The authors have not yet tested if these techniques work when a model is trained from scratch.
*   **Static Thinking:** The current implementation does not dynamically predict when to start or end a rationale; it uses fixed lengths.
*   **Faithfulness:** There is no guarantee that the generated rationales accurately represent the model's actual internal processing.
