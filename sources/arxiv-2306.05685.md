---
id: arxiv:2306.05685
type: paper
title: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)
url: https://arxiv.org/abs/2306.05685
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

# Summary: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

### Core Problem
Traditional LLM benchmarks (e.g., MMLU, HELM) primarily utilize closed-ended questions with short responses to measure core capabilities. However, these metrics often fail to correlate with human preferences in open-ended, multi-turn conversational scenarios. Because human evaluation is slow and expensive, there is a critical need for a scalable, automated method to evaluate how well LLM-based chat assistants align with human preferences.

### Method
The authors propose "LLM-as-a-judge," using strong LLMs (specifically GPT-4) as surrogates for human evaluators. The methodology is implemented through two new benchmarks and a structured judging framework:

**1. Benchmarks**
*   **MT-bench:** A set of 80 high-quality multi-turn questions across eight categories: writing, roleplay, extraction, reasoning, math, coding, STEM, and humanities.
*   **Chatbot Arena:** A crowdsourced platform where users engage in anonymous "battles" between two models and vote for the preferred response.

**2. LLM-as-a-Judge Variations**
*   **Pairwise Comparison:** The judge is presented with a question and two answers and must determine which is better or declare a tie.
*   **Single Answer Grading:** The judge assigns an absolute score to a single response.
*   **Reference-guided Grading:** The judge is provided with a reference solution to assist in grading (primarily for math/reasoning).

**3. Bias Mitigation Recipe**
*   **Position Bias:** To counter the tendency to favor the first response, the authors use a "conservative" swapping approach: the judge is called twice with swapped answer orders; a win is only declared if the same answer is preferred in both instances.
*   **Reasoning Failures:** To improve math/reasoning accuracy, the authors implement **Chain-of-Thought (CoT)** prompting (asking the judge to solve the problem independently first) and **Reference-guided** prompting.
*   **Multi-turn Context:** To prevent referencing errors, the judge is presented with the complete conversation history in a single prompt rather than breaking turns into separate prompts.

### Key Quantitative Results
The study demonstrates that strong LLM judges can closely approximate human judgment:

*   **Agreement Rates:** On MT-bench, GPT-4 achieved an agreement rate with human experts of **85%** (in the non-tie setup S2), which exceeds the agreement rate between humans themselves (**81%**).
*   **Position Bias:** Using default prompts, GPT-4 showed a consistency rate of **65.0%**, while Claude-v1 was significantly lower at **23.8%**. Few-shot prompting increased GPT-4's consistency to **77.5%**.
*   **Verbosity Bias:** In a "repetitive list" attack, GPT-4 had a failure rate of **8.7%**, whereas GPT-3.5 and Claude-v1 both failed **91.3%** of the time.
*   **Math Grading:** GPT-4's failure rate on 10 math questions dropped from **14/20** (default) to **6/20** (CoT) and further to **3/20** (reference-guided).
*   **Model Performance (MT-bench Score):**
    *   GPT-4: **8.99**
    *   GPT-3.5: **7.94**
    *   Vicuna-13B: **6.39**
    *   LLaMA-13B: **2.61**

### Metrics
The authors define **Agreement** as the probability that two randomly selected judges (of the same or different types) agree on a randomly selected question. Two setups are used:
*   **S1:** Includes non-tie, tie, and inconsistent votes (inconsistent votes are counted as ties).
*   **S2:** Includes only non-tie votes.

### Stated Limitations
*   **Biases:** LLM judges are susceptible to **position bias** (favoring specific positions), **verbosity bias** (favoring longer responses), and potential **self-enhancement bias** (favoring their own outputs), though the latter was inconclusive.
*   **Reasoning Gaps:** Judges can be misled by incorrect answers provided in the prompt, even for problems they can solve independently.
*   **Scope:** The evaluation focuses almost exclusively on **helpfulness**, neglecting other critical dimensions such as **safety, honesty, and harmlessness**.
