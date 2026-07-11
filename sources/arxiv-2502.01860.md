---
id: arxiv:2502.01860
type: paper
title: 'SWE-Arena: An Interactive Platform for Evaluating Foundation Models in Software
  Engineering'
url: https://arxiv.org/abs/2502.01860
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-and-winrate-evals
---

# SWE-Arena: An Interactive Platform for Evaluating Foundation Models in Software Engineering

### Core Problem
Existing evaluation frameworks for foundation models (FMs) in software engineering (SE) are insufficient for capturing the iterative, context-heavy nature of real-world development. Static benchmarks (e.g., SWE-bench, BigCodeBench) rely on predefined datasets and ground-truth test cases, failing to account for user-driven, interactive problem-solving. Meanwhile, general-purpose pairwise comparison platforms like Chatbot Arena lack support for multi-round conversational workflows and SE-specific context, and rely on a narrow set of metrics (primarily Elo scores and win rates) that do not reflect the nuances of software engineering tasks.

### Method and Implementation
SWE-Arena is an open-source, interactive platform designed for the end-to-end comparison of FMs through crowd-sourced, pairwise evaluations. The workflow is implemented as follows:

1.  **Initial Query and RepoChat:** Users submit a query. Optionally, they can use **RepoChat**, a feature that accepts a repository URL (e.g., GitHub or GitLab). The platform automatically extracts repository-level metadata—including descriptions, programming languages, issue discussions, commits, and pull/merge requests—and injects this context into the prompt.
2.  **Multi-Round Interaction:** Users engage in anonymous, multi-turn dialogues with two models. This allows for the evaluation of a model's ability to refine solutions based on feedback.
3.  **Guardrails and Constraints:** 
    *   **Filtering:** The platform uses `gpt-5-nano` to automatically filter out prompts that are not relevant to SE tasks.
    *   **Context Management:** To handle context window limits, a First-In, First-Out (FIFO) strategy is used to remove the oldest interactions.
    *   **Timeouts:** Model response times are capped at one minute.
4.  **Voting and Ranking:** Users vote for the superior model. A reassessment feature allows users to modify their votes after reviewing multiple interaction turns to mitigate initial-impression bias.
5.  **Multidimensional Ranking:** The platform aggregates results into a transparent leaderboard using a combination of standard metrics (Elo, Bradley-Terry), advanced graph-based metrics (Eigenvector centrality, PageRank, Newman modularity), and novel SE-specific indices.

### Key Formulas
SWE-Arena introduces two novel metrics to quantify model stability and interaction efficiency:

**Model Consistency Score (MCS):** Measures the percentage of self-play matches where a model produces outputs of similar quality for identical inputs.

$$
MCS = \frac{D}{N} \times 100\%
$$

Where $D$ represents draws against itself and $N$ is the total number of self-play matches.

**Conversation Efficiency Index (CEI):** Evaluates performance relative to the number of interaction rounds ($n_i$) required to reach a conclusion.

$$
CEI = \frac{\sum_{i=1}^{n} \frac{s_i}{n_i}}{\sum_{i=1}^{n} \frac{1}{n_i}}
$$

The score $s_i$ is assigned based on the outcome:

$$
s_{i}=\begin{cases}{1,}&{\text{for win}}\\ {0.3,}&{\text{for draw (both working well)}}\\ {-0.3,}&{\text{for draw (both not working)}}\\ {-1,}&{\text{for loss}}\\ \end{cases}
$$

### Quantitative Results
The provided text focuses on the architectural design and metric definitions of the platform rather than reporting specific benchmark scores for individual models. It specifies a technical constraint that the maximum response time for any model is capped at **one minute**.

### Stated Limitations and Future Work
The authors identify several areas for improvement and future expansion:
*   **Context Management:** The current FIFO strategy for context overflow is basic; the authors propose integrating advanced context compression techniques like LongRope or SelfExtend.
*   **Infrastructure:** The platform currently lacks support for complex SE tasks requiring web browsing or API integration.
*   **Scope:** There is a need to expand coverage to include multimodal foundation models and domain-specific models.
*   **Analysis:** Future work involves analyzing user-submitted requests to establish specialized sub-leaderboards for specific domains, such as debugging or requirement refinement.
