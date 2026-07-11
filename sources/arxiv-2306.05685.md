---
id: arxiv:2306.05685
type: paper
title: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)
url: https://arxiv.org/abs/2306.05685
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-and-winrate-evals
---

# Summary: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

### Core Problem
Traditional LLM benchmarks (e.g., MMLU, HELM) primarily evaluate core capabilities using closed-ended questions with short, automatically verifiable answers. However, these benchmarks fail to capture human preferences in open-ended, multi-turn conversational tasks. Consequently, a model may score well on traditional benchmarks but be perceived as less useful by humans. The authors seek a robust, scalable, and automated method to evaluate LLM alignment with human preferences.

### Method
The researchers propose "LLM-as-a-judge," using strong LLMs (specifically GPT-4) as surrogates for human evaluators.

#### 1. Benchmark Construction
*   **MT-Bench:** A set of 80 high-quality multi-turn questions across eight categories: writing, roleplay, extraction, reasoning, math, coding, STEM, and humanities.
*   **Chatbot Arena:** A crowdsourced platform where users engage in anonymous "battles" between two chatbots and vote for the preferred response.

#### 2. LLM-as-a-Judge Implementation
The authors explore three judging variations:
*   **Pairwise Comparison:** The judge determines which of two answers is better or declares a tie.
*   **Single Answer Grading:** The judge assigns a score (1–10) to a single response.
*   **Reference-Guided Grading:** The judge uses a provided reference solution to grade the response.

#### 3. Bias Mitigation Recipe
To address identified limitations, the following steps are implemented:
*   **Position Bias:** To mitigate the tendency to favor the first response, the authors use a conservative approach: swapping the order of answers and calling the judge twice. A win is only declared if the same answer is preferred in both orders; otherwise, it is a tie.
*   **Math/Reasoning Failures:** To prevent the judge from being misled by incorrect answers, they employ **Chain-of-Thought (CoT)** prompting (asking the judge to solve the problem independently first) or **Reference-Guided** prompting (providing a gold-standard answer).
*   **Multi-turn Context:** To avoid referencing errors, the judge is presented with the complete conversation history in a single prompt rather than breaking turns into separate prompts.

### Key Quantitative Results
*   **Human Agreement:** GPT-4 achieved an agreement rate with human experts exceeding $80\%$. In the "S2" setup (excluding ties) on MT-bench, GPT-4's agreement with humans reached $85\%$, surpassing the human-human agreement rate of $81\%$.
*   **Position Bias:** In zero-shot tests, GPT-4 showed a consistency rate of $65.0\%$. This increased to $77.5\%$ when using a few-shot judge.
*   **Math Grading Accuracy:** For 10 math questions, the failure rate (where GPT-4 called an incorrect answer correct) dropped from $14/20$ (Default) $\rightarrow$ $6/20$ (CoT) $\rightarrow$ $3/20$ (Reference-guided).
*   **Dataset Scale:** The study utilized 3K expert votes for MT-bench and 30K crowdsourced conversations for Chatbot Arena.

### Stated Limitations
*   **Position Bias:** LLMs frequently favor the response presented first.
*   **Verbosity Bias:** Judges tend to favor longer, more verbose responses, even if they are repetitive or lack additional quality.
*   **Self-Enhancement Bias:** There is a potential tendency for judges to favor their own outputs, though the authors found this effect inconclusive due to limited data.
*   **Reasoning Constraints:** LLM judges can be misled by the context of the answers they are grading, leading to arithmetic or logical errors even if the judge can solve the problem in isolation.
*   **Scope:** The current framework focuses on "helpfulness" and largely neglects safety metrics, such as honesty and harmlessness.
