---
id: arxiv:2306.05685
type: paper
title: Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
url: https://arxiv.org/abs/2306.05685
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

**Core Problem**
The rapid deployment of LLM-based chat assistants has revealed a critical misalignment between conventional evaluation benchmarks and actual user preferences. Standard benchmarks (e.g., MMLU, HELM) measure closed-ended, core-knowledge capabilities but fail to assess alignment with human preferences in open-ended, multi-turn dialogues. Consequently, models with strong benchmark scores often generate responses that humans find unhelpful or misaligned. While human evaluation remains the gold standard for conversational utility, it is prohibitively slow and expensive, creating an urgent need for a scalable, automated evaluation framework.

**Method/Recipe Step by Step**
To automate preference assessment, the authors propose a systematic LLM-as-a-judge protocol validated through two benchmarks: MT-bench (80 expert-crafted multi-turn questions across eight categories) and Chatbot Arena (a crowdsourced anonymous battle platform). The evaluation recipe proceeds as follows:
1. **Response Generation:** Target models generate answers to MT-bench prompts or unrestricted Arena queries.
2. **Judge Configuration:** An LLM judge evaluates outputs using one of three modes: pairwise comparison (selecting the superior response), single-answer grading (assigning a 1–10 score), or reference-guided grading (comparing against a ground-truth solution).
3. **Context Preservation:** For multi-turn evaluations, the complete conversation history is presented in a single prompt to prevent contextual referencing errors.
4. **Bias Mitigation:** Response positions are swapped and evaluated twice. A win is declared only if both orders agree (conservative voting); otherwise, it is counted as a tie.
5. **Human Calibration:** LLM judgments are compared against 58 expert labelers (~3K MT-bench votes) and crowdsourced users (~3K sampled from 30K Arena conversations).
6. **Metric Computation:** Inter-judge agreement and average win rates are calculated across all model pairs.

**Key Formulas in LaTeX**
The study defines inter-judge agreement as the probability that two randomly selected, non-identical judges from distinct populations agree on a randomly selected question. When calculating agreement against a human-majority baseline, partial credit is assigned for tied human votes: if human votes split equally between two options and the LLM judge selects one, the agreement is computed as:
$$ \text{Agreement} = \frac{1}{2} $$
Average win rates are derived by excluding tie votes to establish a binary preference baseline.

**Key Quantitative Results**
GPT-4 demonstrates the highest alignment with human preferences, achieving an 85% agreement rate with human experts on MT-bench (non-tie votes), which matches the human-human agreement baseline of 81%. On Chatbot Arena, GPT-4 pairwise and single-answer grading achieve 87% and 85% agreement with crowdsourced humans, respectively. Position bias analysis shows GPT-4 maintains 65.0% consistency when response order is swapped, improving to 77.5% with few-shot prompting, whereas Claude-v1 and GPT-3.5 exhibit severe first-position bias (75.0% and 50.0%, respectively). Under a verbosity attack using repetitive list expansions, GPT-4 demonstrates a failure rate of only 8.7%, compared to 91.3% for weaker models. For math and reasoning tasks, default grading yields a 70% failure rate (14/20), which drops to 30% (6/20) with Chain-of-Thought prompting and further to 15% (3/20) using reference-guided evaluation. Category-wise analysis shows GPT-4 significantly outperforms open models in STEM (76.6% win rate) and humanities (72.2%), while Vicuna-13B struggles in reasoning (20.1%) and math (18.0%).

**Stated Limitations**
The framework acknowledges several constraints. LLM judges exhibit position bias, verbosity bias, and potential self-enhancement bias, though the latter remains statistically inconclusive due to limited data and the difficulty of controlled stylistic rephrasing. Grading math and reasoning questions remains vulnerable to contextual contamination, where judges adopt the flawed logic of provided answers despite possessing independent solving capabilities. The evaluation primarily targets helpfulness, deliberately neglecting safety, honesty, and harmlessness metrics. Furthermore, the framework aggregates distinct dimensions like accuracy, relevance, and creativity into a single scalar score, obscuring nuanced trade-offs. Finally, while fine-tuning open models (e.g., Vicuna-13B) as judges shows promise, it currently suffers from weaker instruction-following and higher output formatting errors compared to proprietary models.
