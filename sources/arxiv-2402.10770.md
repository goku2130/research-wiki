---
id: arxiv:2402.10770
type: paper
title: How Reliable Are Automatic Evaluation Methods for Instruction-Tuned LLMs?
url: https://arxiv.org/abs/2402.10770
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

**Core Problem**
Instruction-tuned Large Language Models are routinely evaluated using automatic metrics to bypass the high cost and time constraints of human annotation. However, the reliability of these methods across diverse tasks, languages, and prompt configurations remains poorly understood. Standard meta-evaluation metrics like Spearman’s $\rho$ and Kendall’s $\tau$ are widely used but fundamentally flawed when applied to human ratings, as they cannot properly handle tied scores, constant rating vectors, or differing metric scales. This study systematically audits the alignment of surface-based, semantic, and LLM-as-a-judge metrics against human judgments to establish when and how automatic evaluations should be trusted.

**Methodology (Step-by-Step)**
1. **Data Preparation:** The Cleaned Alpaca Dataset was used for training. For testing, 20 tasks were sampled from Natural Instructions v2 (15 instances each, 300 total), covering classification, reasoning, and free-form generation. All data was translated to Swedish via GPT-3.5-turbo, yielding monolingual (English, Swedish) and bilingual (English-Swedish) training sets.
2. **Model Instruction Tuning:** Three base models (LLaMA2-7b, LLaMA2-13b, GPT-SW3-6.7b) were fine-tuned on each training set using Alpaca-style prompts and DeepSpeed, producing 12 experimental configurations.
3. **Human Annotation:** Three bilingual annotators rated model outputs on naturalness, relatedness, and correctness using a 1–3 Likert scale. Majority votes established the gold standard, yielding substantial inter-annotator agreement.
4. **Automatic Metric Computation:** ROUGE-L (surface overlap), BERTSCORE (semantic similarity), and GPT-4-as-a-judge (with and without gold references in the prompt) were applied to all generated outputs.
5. **Meta-Evaluation via Pairwise Accuracy:** Instead of correlation coefficients, the authors computed Pairwise Accuracy (PA) with tie calibration. A metric-specific threshold $\epsilon$ was derived from the full dataset to classify score differences below $\epsilon$ as ties, neutralizing scale disparities and tie-preference biases.

**Key Formulas**
The meta-evaluation centers on Pairwise Accuracy, defined as the proportion of correctly ranked pairs, including accurately predicted ties. Tie calibration employs a threshold $\epsilon$ such that any score pair $(s_i, s_j)$ satisfying $|s_i - s_j| < \epsilon$ is treated as a tie. BERTSCORE computes semantic overlap via contextual embedding cosine similarity: $\text{BERTScore} = \cos(\vec{e}_{\text{pred}}, \vec{e}_{\text{ref}})$. ROUGE-L measures surface similarity as the length of the longest common subsequence between the generated response and reference text.

**Key Quantitative Results**
Across all English tasks, GPT-4-gold achieved the highest alignment (PA = 0.81), followed by ROUGE-L (0.75), BERTSCORE (0.66), and GPT-4-no-gold (0.62). Performance is highly task-dependent. For short-answer tasks, ROUGE-L (PA = 0.90) closely matches GPT-4-gold (PA = 0.93). For long-answer tasks, alignment degrades sharply: GPT-4-gold (0.58), BERTSCORE (0.54), GPT-4-no-gold (0.50), and ROUGE-L (0.48). Removing gold references causes GPT-4’s PA to drop by 0.25 in short-answer and 0.09 in long-answer tasks, exposing a strong positive bias (aligning with human 1s only 65% of the time versus 88% with references). In cross-lingual settings, ROUGE-L’s PA decreases by 0.074 for Swedish, while GPT-4-no-gold drops by 0.044. Human annotators demonstrated substantial reliability (Kendall’s $\tau$ = 0.74, Fleiss’ $\kappa$ = 0.63).

**Stated Limitations**
The authors note that PA can yield misleadingly high scores when reference vectors are constant or near-constant, as it lacks prior knowledge of the metric’s score range. The study is restricted to English and Swedish, limiting generalizability to typologically distinct or lesser-resourced languages. Only GPT-4 was evaluated as an LLM judge, leaving open how other models might behave. Finally, the analysis prioritizes correctness; naturalness and relatedness exhibited different alignment patterns and were not the primary focus of the meta-evaluation.
