---
id: arxiv:2306.05685
type: paper
title: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
url: https://arxiv.org/abs/2306.05685
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-and-winrate-evals
---

**Core Problem**
The rapid deployment of instruction-tuned and RLHF-aligned LLM chat assistants has exposed a critical evaluation gap. Traditional benchmarks (e.g., MMLU, HELM) measure core capabilities on closed-ended, short-response tasks and fail to differentiate between base models and fine-tuned conversational variants. This creates a fundamental discrepancy between user preference for helpful, multi-turn dialogue and automated benchmark scores. While human evaluation remains the gold standard for measuring alignment with human preferences, it is prohibitively slow, expensive, and unscalable for iterative model development.

**Method/Recipe Step by Step**
To automate preference evaluation, the authors introduce the "LLM-as-a-judge" framework alongside two benchmarks: MT-bench (80 expert-designed multi-turn questions across eight categories) and Chatbot Arena (crowdsourced anonymous battles). The evaluation pipeline proceeds as follows:
1. **Data Generation:** Collect model responses to MT-bench prompts or Arena user queries.
2. **Judge Configuration:** Deploy an LLM judge using one of three modes: pairwise comparison, single-answer grading, or reference-guided grading.
3. **Bias Mitigation:** Address positional bias by swapping answer order and requiring consistent judgments across both permutations; for multi-turn dialogues, present the complete conversation history in a single prompt to prevent referencing errors.
4. **Reasoning Calibration:** For math and reasoning tasks, apply Chain-of-Thought (CoT) prompting or reference-guided grading, where the judge independently solves the problem first, then compares assistant outputs against its own derived reference.
5. **Aggregation:** Compute agreement metrics between LLM judges and human annotators to validate the automated proxy.

**Key Formulas**
The study formalizes evaluation metrics as follows:
- **Agreement Metric:** The probability that two randomly selected, non-identical judges from different types agree on a randomly selected question:
  $$\text{Agreement} = P(\text{Judge}_A \text{ agrees with } \text{Judge}_B \mid \text{random question})$$
- **MT-Bench Scoring:** Using single-answer grading on a 1–10 scale per turn, the composite score is calculated as:
  $$\text{Total Score} = \sum_{q=1}^{80} \sum_{t=1}^{2} \text{Rating}_{q,t}$$
  where $q$ indexes the 80 questions and $t$ indexes the two conversational turns.

**Key Quantitative Results**
Empirical validation demonstrates that strong LLM judges closely approximate human preferences. On MT-bench, GPT-4 achieves an 85% agreement rate with human experts on non-tied votes, surpassing the human-human agreement baseline of 81%. Chatbot Arena data yields a comparable 87% agreement between GPT-4 and crowdsourced voters. Position bias analysis reveals that GPT-4 maintains 65.0% consistency when answer orders are swapped, improving to 77.5% with few-shot examples, whereas Claude-v1 exhibits a 75.0% bias toward the first position. Verbosity bias is significantly mitigated in GPT-4, which fails the "repetitive list" attack only 8.7% of the time compared to 91.3% for GPT-3.5 and Claude-v1. For mathematical grading, failure rates drop from 14/20 (default) to 6/20 (CoT) and 3/20 (reference-guided). Category-wise win rates show GPT-4 leading in STEM (76.6%) and Humanities (72.2%), while fine-tuned models like Vicuna-13B lag significantly in reasoning (20.1%) and math (18.0%).

**Stated Limitations**
The authors acknowledge several constraints. The framework prioritizes helpfulness while largely neglecting safety, honesty, and harmlessness. Multiple evaluation dimensions (accuracy, relevance, creativity) are aggregated into a single scalar metric, obscuring nuanced trade-offs. Despite mitigation strategies, LLM judges exhibit residual position, verbosity, and self-enhancement biases. Furthermore, even state-of-the-art LLMs demonstrate limited capability in grading basic math and reasoning questions, often adopting incorrect logic from the provided answers. The authors note that while fine-tuning open models as judges shows promise, it requires extensive preference data and careful prompt engineering to match closed-source performance.
