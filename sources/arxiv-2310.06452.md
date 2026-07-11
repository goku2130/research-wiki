---
id: arxiv:2310.06452
type: paper
title: Understanding the Effects of RLHF on LLM Generalisation and Diversity
url: https://arxiv.org/abs/2310.06452
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Summary: Understanding the Effects of RLHF on LLM Generalisation and Diversity

### Core Problem
The authors investigate the impact of the Reinforcement Learning from Human Feedback (RLHF) pipeline on two critical properties of Large Language Models (LLMs): **out-of-distribution (OOD) generalisation** and **output diversity**. While RLHF is widely used to align models with human preferences, the specific contributions of its stages—supervised fine-tuning (SFT), reward modelling (RM), and reinforcement learning (RL)—to these properties were previously under-explored. The study aims to determine if there is an inherent tradeoff between a model's ability to generalise to new distributions and its ability to generate diverse outputs.

### Method and Recipe
The researchers conducted their analysis across two tasks: text summarisation (using a filtered TL;DR Reddit dataset) and instruction following (using the AlpacaFarm framework). They primarily used the LLaMa 7B base model, with additional experiments on OPT models of various scales.

**The fine-tuning pipeline consisted of four compared configurations:**
1.  **Supervised Fine-Tuning (SFT):** The base model is trained on reference input-output pairs using cross-entropy loss.
2.  **Reward Modelling (RM):** A scalar head is added to the base model, which is then trained on pairs of outputs to predict human preferences.
3.  **RLHF:** The SFT model is further fine-tuned using Proximal Policy Optimization (PPO). The policy is optimised to maximise the reward from the RM, subject to a KL divergence penalty to prevent the model from drifting too far from the SFT policy.
4.  **Best-of-N (BoN):** As a baseline to disentangle the effect of the RM from the RL optimisation, $N=16$ samples are drawn from the SFT model, and the RM selects the highest-scoring output.

**Evaluation Metrics:**
*   **Generalisation:** GPT-4 was used as a simulated human evaluator to calculate the "Preference vs Reference" (PvR) win rate. Generalisation was measured by comparing in-distribution (ID) performance against OOD test sets (e.g., CNN/DailyMail for summarisation and Sequential Instructions for instruction following).
*   **Diversity:** The authors measured syntactic (Expectation-Adjusted Distinct N-grams - EAD), semantic (Sentence-BERT cosine similarity), and logical (NLI diversity) metrics. They evaluated both **per-input diversity** (variance of outputs for a single prompt) and **across-input diversity** (variance of outputs across different prompts).

### Key Formulas
The final reward function used during the RLHF stage is defined as:

$$
R(x,y) = RM_{\theta_{RM}}(x,y) - \beta_{KL} D_{KL}(\pi_{\theta_{RL}}(y|x) || \pi_{\theta_{SFT}}(y|x))
$$

where $\beta_{KL}$ is the hyperparameter controlling the KL penalty (set to $0.05$).

Diversity is operationalised as:
*   **Per-Input Diversity:** $\text{PerInputDiversity}_D(\pi) := \frac{1}{N} \sum_{i=1}^N D(O^\pi_i)$
*   **Cross-Input Diversity:** $\text{CrossInputDiversity}_D(\pi) := D(\{O^\pi_i[1]\}_{i=1}^N)$

### Key Quantitative Results
*   **Generalisation:** RLHF consistently improved both ID and OOD performance compared to SFT. In the instruction-following task, RLHF demonstrated significantly better generalisation on the "harder" Sequential Instructions OOD task than SFT. In summarisation, the performance ordering was $\text{Bo16} > \text{RLHF} > \text{SFT}$.
*   **Diversity:** RLHF caused a substantial decrease in per-input diversity compared to SFT. Furthermore, RLHF reduced across-input diversity, providing empirical evidence of "mode collapse," where the model produces similar styles of text regardless of the input.
*   **KL Penalty Impact:** Sweeping the $\beta_{KL}$ coefficient revealed that higher penalties actually resulted in *less* output diversity and generally worse performance, indicating that the KL penalty cannot be used to trade off diversity for generalisation.
*   **Generalisation Gap:** While RLHF performs best in absolute terms OOD, the "generalisation gap" (ID minus OOD performance) was similar between SFT and BoN in summarisation, suggesting the RM generalises well, but the final performance is limited by the underlying SFT model's generalisation.

### Stated Limitations
The authors note that the study provides empirical results without a theoretical explanation for the observed tradeoff. The analysis was limited to a few base models and tasks and only compared SFT, RLHF, and BoN, ignoring other preference learning methods. Finally, the study relied on automatic metrics (GPT-4) rather than direct human evaluation, which may not be a perfect proxy for human judgement.
