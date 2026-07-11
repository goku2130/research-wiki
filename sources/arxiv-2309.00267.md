---
id: arxiv:2309.00267
type: paper
title: 'RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI
  Feedback'
url: https://arxiv.org/abs/2309.00267
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

# RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback

### Core Problem
Reinforcement Learning from Human Feedback (RLHF) is highly effective for aligning Large Language Models (LLMs) with human preferences, but it is limited by the high cost and scalability constraints of gathering high-quality human preference labels. This research investigates whether Reinforcement Learning from AI Feedback (RLAIF)—where an off-the-shelf LLM generates the preference labels—can serve as a viable, cost-effective alternative to RLHF without sacrificing performance.

### Methodology

#### 1. Canonical RLAIF
The authors implement a pipeline to replace human annotators with an AI labeler (PaLM 2 Large):
1.  **Preference Generation:** The AI labeler is prompted with a preamble, optional few-shot exemplars, and a pair of candidate responses. To mitigate **position bias**, the model performs two inferences with the candidates reversed and averages the resulting preference distributions.
2.  **Label Enhancement:** To improve alignment, the authors employ **Chain-of-Thought (CoT)** reasoning, where the LLM first generates a rationale before selecting the preferred response.
3.  **Reward Model (RM) Training:** A reward model is trained using cross-entropy loss on the soft labels (probability distributions) provided by the AI labeler.
4.  **Policy Optimization:** The final policy is trained using a modified version of REINFORCE with a baseline, optimizing the reward provided by the RM.

#### 2. Direct-RLAIF (d-RLAIF)
To avoid the time-consuming process of RM training and the issue of "RM staleness" (where the RM becomes out-of-distribution as the policy evolves), the authors introduce **d-RLAIF**. In this setup, the off-the-shelf LLM provides rewards directly during the RL phase by rating generations on a scale of 1 to 10.

### Key Formulas

**AI Labeler Alignment ($z_{acc}$):**
Measures the accuracy of AI-labeled preferences relative to human preferences:

$$
z_{acc} = \frac{1}{D} \sum_{i=1}^{D} \mathbb{1}[\mathop{\text{arg max}}_{j} P_{i,j}^{AI} = p_{i}^{H}]
$$

where $D$ is the dataset size, $P^{AI}$ is the matrix of soft AI preferences, and $p^{H}$ is the vector of human preferences.

**d-RLAIF Reward Score:**
The reward $s(y|x)$ is calculated as a weighted average of the likelihood of score tokens (1–10):

$$
s(y|x) = \frac{\sum_{i=1}^{10} i P(i|y,x)}{\sum_{i=1}^{10} P(i|y,x)}
$$

This score is subsequently normalized to the range $[-1, 1]$.

**RL Optimization Objective:**
The policy $\pi_{\theta}^{RL}$ is optimized to maximize reward while penalizing deviation from the SFT policy $\pi^{SFT}$ via KL divergence:

$$
J(\theta) = \mathbb{E}_{y\sim\pi_{\theta}(\cdot|x)} \left[ (1-\beta)r_{\phi}(y|x) - \beta D_{KL}(\pi_{\theta}^{RL}(y|x) || \pi^{SFT}(y|x)) \right]
$$

### Key Quantitative Results

The study evaluated three tasks: summarization, helpful dialogue, and harmless dialogue.

*   **Win Rates (vs. SFT Baseline):**
    *   **Summarization:** RLAIF (71%) and RLHF (73%) performed similarly.
    *   **Helpful Dialogue:** RLAIF (63%) and RLHF (64%) performed similarly.
    *   **Harmless Dialogue:** RLAIF achieved a higher harmless rate (88%) compared to RLHF (76%) and SFT (64%).
*   **Head-to-Head:** RLAIF and RLHF were equally preferred by humans (approx. 50% win rate) for summarization and helpful dialogue.
*   **Self-Improvement:** "Same-size RLAIF" (where the labeler is the same size as the policy) still outperformed SFT with a 68% win rate in summarization. d-RLAIF achieved a 66% win rate over SFT in helpful dialogue using the exact same model checkpoint for both the policy and the reward provider.
*   **Cost Efficiency:** AI preference labeling is estimated to be over **10x cheaper** than human annotation ($\approx \$0.06$ vs. $\approx \$0.67$ per example).
*   **Alignment:** The best AI labeler alignment for summarization was 78.0%, which is comparable to human inter-annotator agreement (73–77%).

### Stated Limitations
*   **Fluency and Coherence:** Qualitative analysis revealed that RLAIF summaries were sometimes less fluent than RLHF summaries, occasionally containing run-on sentences or repeating phrases.
*   **Hallucinations:** While RLHF was observed to hallucinate in some cases where RLAIF did not, the difference in overall accuracy, coverage, and coherence was not statistically significant.
*   **Prompting Sensitivity:** In-context learning (few-shot prompting) actually decreased alignment for summarization and helpfulness tasks, though it helped with harmlessness.
*   **Self-Consistency:** Sampling multiple CoT rationales with temperature $T > 0$ strictly degraded AI labeler alignment.
