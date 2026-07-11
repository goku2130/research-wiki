---
id: arxiv:2306.05685
type: paper
title: Language Models are Good Evaluators but Need Careful Prompting
url: https://arxiv.org/abs/2306.05685
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-and-winrate-evals
---

# Summary: Language Models are Good Evaluators but Need Careful Prompting

### Core Problem
Traditional LLM benchmarks (e.g., MMLU, HELM) primarily evaluate core capabilities using closed-ended questions with short, automatically verifiable answers. However, these benchmarks fail to capture human preferences in open-ended, multi-turn conversations, creating a discrepancy between benchmark scores and actual user perceptions of a chatbot's utility. The authors seek a scalable, automated method to evaluate LLM alignment with human preferences.

### Method
The authors propose the **LLM-as-a-judge** framework, using strong LLMs (specifically GPT-4) as surrogates for human evaluators. To validate this, they introduced two benchmarks:
1.  **MT-bench:** A set of 80 high-quality multi-turn questions across eight categories: writing, roleplay, extraction, reasoning, math, coding, STEM, and humanities.
2.  **Chatbot Arena:** A crowdsourced platform where users engage in anonymous "battles" between two models and vote for the preferred response.

#### LLM-as-a-Judge Implementation
The framework employs three primary evaluation variations:
*   **Pairwise Comparison:** The judge compares two answers to one question and determines a winner or a tie.
*   **Single Answer Grading:** The judge assigns an absolute score (1–10) to a single response.
*   **Reference-guided Grading:** The judge is provided with a reference solution to guide the evaluation, particularly for math and reasoning.

#### Mitigation of Biases
To address identified limitations, the authors implemented several "recipes":
*   **Position Bias:** Mitigated by swapping the order of responses and calling the judge twice; a win is only declared if the same model is preferred in both orders.
*   **Reasoning/Math Failures:** Mitigated via **Chain-of-Thought (CoT)** prompting (asking the judge to solve the problem independently first) or **Reference-guided** prompting.
*   **Multi-turn Context:** To prevent referencing errors, the judge is presented with the complete conversation history in a single prompt rather than breaking turns into separate prompts.

### Key Quantitative Results
*   **Human Agreement:** On MT-bench, GPT-4 achieved an agreement rate with human experts of **85%** (non-tie setup), which exceeds the agreement rate between humans themselves (**81%**).
*   **Chatbot Arena:** GPT-4 (single-answer grading) matched human preferences with an agreement rate of **85%** (non-tie).
*   **Position Bias:** In zero-shot tests, GPT-4 showed a consistency rate of **65.0%**, which increased to **77.5%** using a few-shot judge.
*   **Math Grading Accuracy:** For GPT-4, the failure rate (marking an incorrect answer as correct) dropped from **14/20** (default) $\rightarrow$ **6/20** (CoT) $\rightarrow$ **3/20** (Reference-guided).
*   **Verbosity Bias:** Under a "repetitive list" attack, GPT-4 was significantly more resilient than Claude-v1 and GPT-3.5, which both had failure rates of **91.3%**.

### Technical Examples
The authors highlight the difficulty of grading reasoning tasks, such as the inequality $|x + 5| < 10$. While GPT-4 can solve this independently (finding 19 integers), it was misled by a verbose but incorrect assistant response claiming 20 integers, illustrating the need for reference-guided prompts.

### Stated Limitations
*   **Scope of Evaluation:** The study focuses on "helpfulness" but neglects **safety, honesty, and harmlessness**.
*   **Metric Granularity:** Dimensions such as accuracy, relevance, and creativity are collapsed into a single helpfulness metric.
*   **Biases:** Despite mitigations, LLM judges remain susceptible to position bias (favoring the first response) and verbosity bias (favoring longer responses).
*   **Reasoning:** LLMs can be misled by the context of the answers they are grading, even when they possess the capability to solve the problem in isolation.
