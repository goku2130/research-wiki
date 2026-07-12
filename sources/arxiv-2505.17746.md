---
id: arxiv:2505.17746
type: paper
title: 'Fast Quiet-STaR: Thinking Without Thought Tokens'
url: https://arxiv.org/abs/2505.17746
retrieved: '2026-07-12'
maturity: comprehensive
topic: rl-for-reasoning
---

# Fast Quiet-STaR: Thinking Without Thought Tokens

## Core Problem
Quiet-STaR (Quiet Self-Taught Reasoner) improves the reasoning capabilities of Large Language Models (LLMs) by generating token-level "thought traces" before predicting the next token. However, this mechanism introduces substantial inference overhead. Generating thought tokens for every single token significantly increases the Time-to-First-Token (TTFT) and overall generation latency. Furthermore, simply reducing the number of thought tokens in a standard Quiet-STaR model leads to a noticeable drop in reasoning accuracy, creating a trade-off between efficiency and performance.

## Method
Fast Quiet-STaR addresses this by employing a curriculum learning strategy to compress thought traces and a reinforcement learning (RL) phase to internalize reasoning into a standard Next Token Prediction (NTP) setting.

### 1. Quiet-STaR Baseline
The framework builds on the Quiet-STaR process:
*   **Think:** For a sequence $X$, the model generates a thought $T_i$ of length $n-1$ enclosed by `<|start_of_thought|>` and `<|end_of_thought|>` tokens.
*   **Talk:** A learnable interpolation mechanism uses a shallow MLP to compute a weight $w$, mixing the base logits and thought-conditioned logits:

$$
\log p_i^{\text{talk}} = w_i \log p_i^{\text{base}} + (1 - w_i) \log p_i^{\text{thought}}
$$

*   **Learn:** The model is optimized using the REINFORCE algorithm. The reward $r_j$ is the difference between the log-likelihood of the next $m$ ground-truth tokens given a specific rationale $T_j$ and the mean log-likelihood across all sampled rationales:

$$
r_j = \log p_{j:j+m}^{\text{talk}}(X_{j+1:j+m+1}) - \log \bar{p}_{j:j+m}^{\text{talk}}(X_{j+1:j+m+1})
$$

    The parameters $\theta$ are updated via:

$$
\nabla_{\theta} \mathcal{L}_j^{\text{REINFORCE}} = - r_j \cdot \nabla_{\theta} \log p_{\theta}(T_j \mid [X_{:j}; <|\text{start\_of\_thought}|>])
$$

### 2. Fast Quiet-STaR Curriculum Learning
To minimize thought tokens without losing performance, the authors implement a multi-stage "easy-to-hard" training strategy. The model is progressively trained to generate more concise and abstract reasoning:
*   **Stage 1:** 16 thought tokens and 8 ahead tokens (16-8).
*   **Stage 2:** 12 thought tokens and 4 ahead tokens (12-4).
*   **Stage 3:** 8 thought tokens and 4 ahead tokens (8-4).

### 3. Fast Quiet-STaR NTP
To eliminate explicit thought tokens during inference entirely, the model is transitioned to the NTP paradigm using RL fine-tuning:
*   **Initialization:** An NTP model is initialized from the Fast Quiet-STaR 8-4 checkpoint.
*   **Reward Calculation:** The reward is the difference between the negative log-likelihood (NLL) of the Fast Quiet-STaR 8-4 model ($\mathcal{L}_{\text{Fast Quiet-STaR}}$) and the NTP model ($\mathcal{L}_{\text{Fast Quiet-STaR-NTP}}$):

$$
r_j = \mathcal{L}_{\text{Fast Quiet-STaR}} - \mathcal{L}_{\text{Fast Quiet-STaR-NTP}}
$$

*   **Optimization:** The model is updated to emulate the prediction quality of the thinking model without generating explicit tokens:

$$
\nabla_{\theta} \mathcal{L}_j^{\text{REINFORCE}} = - r_j \cdot \nabla_{\theta} \log p_{\theta}(x_j \mid X_{:j})
$$

## Key Quantitative Results
Experiments were conducted using Mistral 7B and Qwen2.5 7B on datasets including PIQA, SIQA, CommonsenseQA, and GSM8K.

*   **Accuracy Gains:** Fast Quiet-STaR NTP improved average accuracy by **9% on Mistral 7B** and **5.7% on Qwen2.5 7B** compared to the original pre-trained models while maintaining the same inference latency.
*   **Efficiency vs. Quiet-STaR:** For Mistral 7B, Fast Quiet-STaR (8 tokens) outperformed the Quiet-STaR (16 tokens) variant by **1.8%** while reducing inference time to **41.3%**.
*   **Latency Reduction:** Fast Quiet-STaR NTP achieved only **6%** of the end-to-end generation time of Quiet-STaR 16-8 (for a 256-prefix, 128-generation length scenario).
*   **Complex Reasoning (GSM8K):** Using Chain-of-Thought (CoT) with majority voting (maj@6), Fast Quiet-STaR NTP increased accuracy from **43.3% to 52.4%** over the pre-trained model.
*   **Training Cost:** The entire pipeline required only **0.5M tokens** and took **54 minutes** on 8 H800 GPUs.

## Limitations
The authors state two primary limitations:
1.  The evaluation focused primarily on mathematical and logical reasoning; generalization to other domains remains unvalidated.
2.  The proposed method is specifically designed for the Quiet-STaR reasoning framework and may not apply to other paradigms.
