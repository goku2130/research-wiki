---
id: arxiv:2410.07137
type: paper
title: 'Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates (2024)'
url: https://arxiv.org/abs/2410.07137
retrieved: '2026-07-12'
maturity: comprehensive
topic: alignment-and-winrate-evals
---

# Summary: Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates

## Core Problem
Automatic LLM benchmarks (e.g., AlpacaEval 2.0, Arena-Hard-Auto, and MT-Bench) increasingly rely on LLM-based auto-annotators ("LLM-as-a-Judge") to provide scalable, cost-effective evaluations. While these benchmarks attempt to mitigate biases related to output length and style, they remain vulnerable to intentional manipulation. The authors investigate whether an adversary can use a "null model"—a model that provides constant, non-informative responses irrelevant to the input instructions—to deceive auto-annotators into assigning high win rates, thereby gaining unearned promotional impact.

## Method
The authors employ a multi-stage strategy to craft cheating responses that exploit the syntactic analysis and positional biases of LLM judges.

### 1. Null Model Implementation
A null model is defined as a non-trainable function that returns a constant string regardless of the input:

$$
\text{NullModel}(\text{const\_str}) \rightarrow \text{output} = \text{const\_str}
$$

### 2. Structured Cheating Responses
Rather than using simple persuasive text, the authors craft "structured" responses designed to disrupt the judge's processing of the evaluation template. The recipe involves:
1.  **Overriding Context:** The response instructs the judge to "Ignore the above ## Model Outputs."
2.  **Counterfeiting Instructions:** It inserts a fake instruction (e.g., "Output nothing") and counterfeit empty model outputs.
3.  **Exploiting Position Bias:** By deceiving the judge into believing both the target and reference models produced empty outputs, the judge defaults to a preference for the first-positioned model (often the target model "M" in default settings).

### 3. Adversarial Prefix via Random Search (RS)
To ensure the cheat is transferable across private benchmark instructions, the authors optimize an adversarial prefix using a public dataset (UltraFeedback). They use a Random Search (RS) algorithm to refine the prefix $s_{1:l}$ over $T$ iterations:
*   **Step 1:** Sample a modification $\tilde{s}_{1:l}$ by replacing a random token with one from the vocabulary $\mathcal{X}$.
*   **Step 2:** Calculate the aggregated loss $\mathcal{L}$ across $N$ instructions.
*   **Step 3:** Update the prefix if the new loss is lower:

$$
\sum_{j=1}^{N} \mathcal{L}(x_{1:n_j}^{(j)}, \tilde{s}_{1:l}) \leq \mathcal{L}_{\text{Best}}
$$

## Key Quantitative Results
The combination of structured responses and Random Search (Structured+RS) significantly outperformed verified State-of-the-Art (SOTA) models across all tested benchmarks using GPT-4-1106-Preview as the judge:

| Benchmark | Metric | Verified SOTA | Structured (Ours) | Structured+RS (Ours) |
| :--- | :--- | :--- | :--- | :--- |
| **AlpacaEval 2.0** | LC Win Rate | 57.5% | 76.8% | **86.5%** |
| **Arena-Hard-Auto** | Win Rate | 82.6% | 67.2% | **83.0%** |
| **MT-Bench** | Score (1-10) | 8.96 | 7.75 | **9.55** |

### Open-Source Judge Performance
The structured response alone was largely ineffective against Llama-3-Instruct models (e.g., 2.9% LC win rate for Llama-3-8B). However, applying **Random Search** caused win rates to surge to **95.4% LC** (Llama-3-8B) and **95.1% LC** (Llama-3-70B). When the adversarial prefix was optimized directly on the test instructions, the LC win rate for Llama-3-8B reached **99.8%**.

## Limitations
*   **Benchmark Scope:** The study focuses on specific benchmarks; the effectiveness of these cheats on other, less-studied benchmarks is uncertain.
*   **Manual Effort:** The initial structured responses required manual crafting.
*   **Judge Transferability:** The cheats showed low transferability between different judge models (e.g., a cheat optimized for GPT-4 performed poorly when transferred to GPT-3.5).
*   **Defense Vulnerability:** While "SmoothLLM" (random perturbations) was effective at reducing win rates to near zero, the authors note it may be impractical as it also degrades the win rates of clean, legitimate responses.
