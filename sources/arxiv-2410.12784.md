---
id: arxiv:2410.12784
type: paper
title: 'JudgeBench: A Benchmark for Evaluating LLM-based Judges'
url: https://arxiv.org/abs/2410.12784
retrieved: '2026-07-12'
maturity: comprehensive
topic: llm-as-judge
---

# JudgeBench: A Benchmark for Evaluating LLM-based Judges

**JudgeBench** is a benchmark designed to objectively evaluate the reliability of LLM-based judges, specifically their ability to distinguish between factually and logically correct responses versus incorrect ones in complex domains.

### Core Problem
LLM-based judges are increasingly used as scalable alternatives to human evaluation. However, existing benchmarks for these judges typically measure alignment with crowdsourced human preferences. The authors argue that for complex tasks—such as mathematical proofs or code verification—human annotators often lack the domain expertise to judge correctness and instead mistakenly prioritize stylistic factors (e.g., length or plausibility) over factual accuracy. Consequently, there is a need for a benchmark that prioritizes objective correctness over subjective style.

### Method and Pipeline
The authors propose a hierarchical evaluation framework where a judge must prioritize: (1) instruction following, (2) factual and logical correctness, and (3) stylistic alignment. To implement this, they developed a pipeline to transform existing difficult datasets into challenging response pairs:

1.  **Source Selection:** Questions with ground truth labels and verification algorithms are sourced from challenging datasets: **MMLU-Pro** (Knowledge), **LiveBench** (Reasoning and Mathematics), and **LiveCodeBench** (Coding).
2.  **Response Generation:** A strong LLM (e.g., GPT-4o) is used to sample $k$ responses for each question.
3.  **Correctness Grading:** Each response is graded using the dataset's verification algorithm. To prevent "incorrect" labels caused by minor formatting errors, a secondary LLM (GPT-4o-mini) verifies the solution's correctness regardless of format.
4.  **Pair Construction:** Questions are filtered to retain only those with at least one correct and one incorrect response. These are paired to create a test set where the ground truth preference is objectively determined by correctness.
5.  **Bias Mitigation:** To counter positional bias, each pair is evaluated twice with the response order swapped.
6.  **Verdict Aggregation:** The final decision is determined as follows:
    *   If both trials yield $A > B$, or one yields $A > B$ and the other $A = B$, the aggregate decision is $A > B$.
    *   Inconsistent decisions (e.g., $A > B$ in one trial and $B > A$ in the other) or double ties are marked as incorrect.

### Key Quantitative Results
The final JudgeBench dataset consists of 350 questions: 154 Knowledge, 98 Reasoning, 56 Mathematics, and 42 Coding.

*   **Prompted Judges:** Standard prompted judges struggle significantly. The Vanilla judge (GPT-4o) achieved an overall accuracy of $50.86\%$, barely exceeding random guessing. The Arena-Hard prompt improved this to $56.57\%$.
*   **Fine-tuned Judges:** Most fine-tuned judges performed below the $50\%$ random baseline, with the exception of Skywork's judges (e.g., Skywork-LLaMA-3.1B-70B at $57.43\%$).
*   **Reasoning-Enhanced Models:** Models that scale test-time compute performed best. OpenAI's **o3-mini (high)** achieved the highest overall accuracy at $80.86\%$.
*   **Reward Models:** Specialized reward models generally outperformed LLM-based judges. For example, Skywork-Reward-Gemma-2-27B achieved $64.29\%$ accuracy, comparable to Claude-3.5-Sonnet.
*   **Comparative Difficulty:** JudgeBench is more challenging than prior benchmarks (MT-Bench, LLMEval, FairEval, and LLMBar). The strongest general-purpose model on JudgeBench achieved only $64\%$ accuracy, whereas similar models reached higher saturation on other benchmarks.

### Stated Limitations
*   **Pipeline Bias:** Because GPT-4o was used to generate the response pairs, the dataset may be disproportionately challenging for GPT-4o judges. Ablation studies confirmed that models (like Claude-3.5-Sonnet) perform worse when judging their own generated pairs.
*   **Technical Constraints:** Some fine-tuned judges (PandaLM, JudgeLM) have limited context windows (2048 tokens), necessitating left-truncation of responses, which may impact performance.
*   **Judge Reliability:** Many fine-tuned judges exhibited high rates of inconsistency between trials or produced invalid judgments (e.g., Prometheus2-bgb-8x7b).
