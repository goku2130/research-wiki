---
id: arxiv:2402.14762
type: paper
title: 'MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models
  in Multi-Turn Dialogues'
url: https://arxiv.org/abs/2402.14762
retrieved: '2026-07-12'
maturity: comprehensive
topic: alignment-and-winrate-evals
---

# MT-Bench-101: A Fine-Grained Benchmark for Evaluating LLMs in Multi-Turn Dialogues

### Core Problem
Evaluating the multi-turn dialogue capabilities of Large Language Models (LLMs) remains a significant challenge. Existing benchmarks typically focus on single-turn instructions or provide coarse-grained assessments that fail to capture the complexity and fine-grained nuances of real-world human-chatbot interactions. There is a critical need for a comprehensive, granular framework to identify specific deficiencies in how models handle context, adapt to feedback, and proactively engage users over multiple turns.

### Method and Recipe
The authors developed **MT-Bench-101**, a benchmark structured around a three-tier hierarchical ability taxonomy derived from real dialogue data (ShareGPT, RealChat) and educational psychology.

**1. Taxonomy Construction**
*   **Tier 1 (Overarching Abilities):** 
    *   *Perceptivity:* Accuracy in understanding context.
    *   *Adaptability:* Ability to respond effectively to user feedback.
    *   *Interactivity:* Capacity for proactive engagement.
*   **Tier 2:** Seven detailed abilities.
*   **Tier 3:** 13 distinct tasks, including Context Memory (CM), Anaphora Resolution (AR), Separate Input (SI), Topic Shift (TS), Content Confusion (CC), Content Rephrasing (CR), Format Rephrasing (FR), Self-correction (SC), Self-affirmation (SA), Mathematical Reasoning (MR), General Reasoning (GR), Instruction Clarification (IC), and Proactive Interaction (PI).

**2. Data Generation and Curation**
*   **Generation:** GPT-4 was used to generate over 1,000 samples per task using unique prompts and manually crafted examples across 30 diverse topics (e.g., health, law, finance).
*   **Curation:** Five human annotators screened the data; only samples deemed high-quality by all five annotators were retained.

**3. Evaluation Protocol**
*   **Context:** The benchmark utilizes "golden context" (ground truth history) to ensure fluid dialogues and prevent error propagation.
*   **Judging:** GPT-4 acts as the primary judge, using task-specific scoring guidelines to rate responses on a scale of 1 to 10.
*   **Scoring Metric:** To prevent inflated scores and reflect the reality that one failed turn can compromise a conversation, the benchmark employs a minimum-score-taking metric:

$$
\text{Total Score} = \min(\text{score}_1, \text{score}_2, \dots, \text{score}_n)
$$

where $\text{score}_i$ is the score for the $i$-th turn of the dialogue.

### Key Quantitative Results
*   **Dataset Scale:** MT-Bench-101 consists of 1,388 multi-turn dialogues encompassing 4,208 turns across 13 tasks.
*   **Model Performance:** 21 LLMs were evaluated. GPT-4 was the top performer with an average score of $8.86$, followed by Yi-34B at $8.10$.
*   **Human Alignment:** The agreement between GPT-4 judges and human experts reached $87\%$, surpassing the internal agreement among human experts ($80\%$).
*   **Alignment Techniques:** RLHF/DPO showed marginal effects on multi-turn abilities. For InternLM2-Chat, scores increased by $+0.16$ (7B) and $+0.10$ (20B), while Mistral-7B saw a decrease of $-0.06$.
*   **Task Difficulty:** Mathematical reasoning was identified as the most challenging task, while content confusion and format rephrasing were the least difficult.

### Stated Limitations
*   **Residual Bias:** Despite human curation, residual errors or biases from GPT-4 or annotators may persist.
*   **Data Leakage:** Public availability of the dataset could lead to it being used for training, potentially reducing the benchmark's effectiveness.
*   **Taxonomy Evolution:** As LLMs evolve, the current capability taxonomy may become incomplete.
*   **Usage:** The dataset is intended for research purposes and may not be suitable for commercial use without further verification.
