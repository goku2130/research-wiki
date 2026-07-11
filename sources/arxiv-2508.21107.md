---
id: arxiv:2508.21107
type: paper
title: Learning to Generate Unit test via Adversarial Reinforcement Learning
url: https://arxiv.org/html/2508.21107v2
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

The paper "Learning to Generate Unit test via Adversarial Reinforcement Learning" introduces UTRL, a novel reinforcement learning framework for training Large Language Models (LLMs) to generate high-quality unit tests.

**Core Problem:**
The core problem addressed is the challenge of generating comprehensive and high-quality unit tests for programming tasks. Traditional methods often rely on supervised fine-tuning (SFT) with instruction-unit test pairs, which are costly and labor-intensive to obtain at scale. Existing RL approaches for unit test generation are underexplored, particularly in designing reward functions that can reliably assess unit test quality without ground-truth annotations. Unit tests need to contain functionally valid test cases and cover challenging edge cases to discriminate subtly faulty code implementations.

**Method/Recipe Step by Step (UTRL Framework):**
UTRL iteratively trains two LLMs in an adversarial manner: a unit test generator ($\mathcal{M}_{\text{UT}}$) and a code generator ($\mathcal{M}_{\text{code}}$). The framework operates without requiring explicit instruction-unit test pairs, relying instead on instruction-code solution pairs.

1.  **Initialization:**
    *   Start with an initial code generator LLM ($\mathcal{M}_{\text{code}}$) and an initial unit test generator LLM ($\mathcal{M}_{\text{UT}}$).
    *   Define hyperparameters: $\lambda$ (weight for discrimination reward) and $\tau$ (desired minimum number of test cases per unit test).
    *   Provide a dataset of instruction-code solution pairs $\mathcal{D} = \{(I_k, C_k^*)\}$, where $I$ is the instruction and $C^*$ is the ground-truth code solution.

2.  **Iterative Training Loop (while not converged):**

    *   **Step 1: Training the Unit Test Generator ($\mathcal{M}_{\text{UT}}$):**
        *   For each instruction-code solution pair $(I, C^*) \in \mathcal{D}$:
            *   Generate a unit test $\mathcal{T}$ using $\mathcal{M}_{\text{UT}}$: $\mathcal{T} \sim \mathcal{M}_{\text{UT}}(\cdot | I)$.
            *   Generate a set of code solutions $\mathcal{C}$ using $\mathcal{M}_{\text{code}}$: $\mathcal{C} \sim \mathcal{M}_{\text{code}}(\cdot | I)$.
            *   Compute the reward for $\mathcal{T}$ ($r_{\text{UT}}$) using a weighted sum of discrimination reward ($R_{\text{disc}}$) and validity reward ($R_{\text{valid}}$).
            *   Update $\mathcal{M}_{\text{UT}}$ using an RL update rule (e.g., GRPO) based on $r_{\text{UT}}$.

    *   **Step 2: Training the Code Generator ($\mathcal{M}_{\text{code}}$):**
        *   For each instruction-code solution pair $(I, C^*) \in \mathcal{D}$:
            *   Generate a code solution $C$ using $\mathcal{M}_{\text{code}}$: $C \sim \mathcal{M}_{\text{code}}(\cdot | I)$.
            *   Generate a unit test $\mathcal{T}$ using $\mathcal{M}_{\text{UT}}$: $\mathcal{T} \sim \mathcal{M}_{\text{UT}}(\cdot | I)$.
            *   Compute the reward for $C$ ($r_{\text{code}}$) based on its pass rate over $\mathcal{T}$.
            *   Update $\mathcal{M}_{\text{code}}$ using an RL update rule (e.g., GRPO) based on $r_{\text{code}}$.

**Key Formulas (in LaTeX):**

1.  **Discrimination Reward ($R_{\text{disc}}$):** Evaluates how effectively a unit test $\mathcal{T}$ discriminates LLM-generated code solutions from ground-truth code solutions.
    $$R_{\text{disc}}(\mathcal{T}, \mathcal{C}, C^*) = \frac{1}{|\mathcal{C}|} \sum_{C \in \mathcal{C}} \left[ 1 - \prod_{T \in \mathcal{T}} (1 - \text{Pass}(C, T))^{\text{Pass}(C^*, T)} \right]$$
    where $\text{Pass}(C, T)$ is an indicator function (1 if code $C$ passes test case $T$, 0 otherwise). The term $\text{Pass}(C^*, T)$ filters out functionally invalid test cases (those not passed by the ground-truth code).

2.  **Validity Reward ($R_{\text{valid}}$):** Evaluates the functional validity of test cases in $\mathcal{T}$.
    $$R_{\text{valid}}(\mathcal{T}, C^*, \tau) = \frac{\sum_{T \in \mathcal{T}} \text{Pass}(C^*, T)}{\max(|\mathcal{T}|, \tau)}$$
    where $\tau$ is a hyperparameter for the desired minimum number of test cases, preventing high rewards for unit tests with few trivial cases.

3.  **Unit Test Generator Training Reward ($r_{\text{UT}}$):** A weighted sum of the discrimination and validity rewards.
    $$r_{\text{UT}} = \lambda R_{\text{disc}}(\mathcal{T}, \mathcal{C}, C^*) + (1 - \lambda) R_{\text{valid}}(\mathcal{T}, C^*, \tau)$$
    where $\lambda$ is a hyperparameter weighting the discrimination reward.

4.  **Code Generator Training Reward ($R_{\text{code}}$):** Measures the proportion of functionally valid test cases (from $\mathcal{T}$) that are passed by the generated code $C$.
    $$R_{\mathrm{code}}(C, \mathcal{T}, C^*) = \frac{\sum_{T \in \mathcal{T}} \operatorname{Pass}(C, T) \cdot \operatorname{Pass}(C^*, T)}{\sum_{T \in \mathcal{T}} \operatorname{Pass}(C^*, T)}$$

**Key Quantitative Results and Numbers:**

*   **Best-of-N Improvement:**
    *   Qwen3-4B trained with UTRL yielded code accuracies of **14.9%** (with Qwen3-8B code generator) and **17.3%** (with Qwen3-14B code generator) in best-of-32 sampling.
    *   This is compared to Qwen3-4B trained with SFT, which achieved **11.7%** and **14.0%** respectively.
    *   UTRL-trained Qwen3-4B outperformed GPT-4.1 and GPT-4o in this metric.
    *   Qwen3-14B trained with UTRL achieved even stronger performance: **15.0%** and **17.7%**.
    *   UTRL achieved **4.4%** and **3.2%** higher code accuracies than CURE (an RL baseline) when using Qwen2.5-7B-Instruct.
    *   On LiveCodeBench, UTRL-trained Qwen3-4B achieved **59.9%** (with Qwen3-4B code generator) and **59.3%** (with Qwen3-8B code generator) accuracy, outperforming GPT-4.1.

*   **Unit Test Fidelity (Spearman's correlation with GT unit tests):**
    *   Qwen3-14B trained with UTRL achieved **0.827**.
    *   Qwen3-4B trained with UTRL achieved **0.794**.
    *   UTRL-trained models achieved higher fidelity than SFT-trained models and GPT-4.1.
    *   UTRL achieved a fidelity of **0.593**, surpassing CURE.

*   **Code Generator Effectiveness:**
    *   Qwen3-4B code generator trained with UTRL achieved a pass@1 code accuracy of **15.3%**.
    *   This is "remarkably higher" than baselines (e.g., SFT with ground-truth code solutions yielded **3.6%** pass@1 accuracy).
    *   It was "comparable" to the upper bound of **15.9%** achieved by training to maximize pass rate over GT unit tests.

*   **Effect of Iterative Training:**
    *   In iteration 1, discrimination reward saturated after 50 steps with a gain of about **0.02**.
    *   In iteration 2, the discrimination reward dropped from **0.626** to **0.375** initially, then improved from **0.375** to **0.447**.
    *   Iteration 2 unit tests yielded "superior best-of-N performance" compared to iteration 1, even outperforming GPT-4.1.

*   **Ablation on $\lambda$ (weighting factor for discrimination reward):**
    *   $\lambda = 0.85$ achieved the best unit test quality.
    *   $\lambda = 0.0$ resulted in a validity ratio of **64.3%** but lower Best-of-N improvement.
    *   $\lambda = 1.0$ resulted in a validity ratio of **49.7%** and degraded Best-of-N improvement.

*   **Ablation on Validity Reward:**
    *   UTRL without $R_{\text{valid}}$ resulted in unit tests with "more than 50%" invalid test cases.
    *   UTRL with $R_{\text{valid}}$ resulted in unit tests with "less than 35%" invalid test cases.
    *   UTRL without clipping in $R_{\text{valid}}$ led to models generating "extremely small number of trivial test cases."

**Stated Limitations:**
1.  **Performance Gap:** A performance gap still exists between UTRL-generated unit tests and ground-truth unit tests.
2.  **Domain Specificity:** Experiments were conducted on competitive programming tasks; broader software engineering domains need to be explored.
3.  **Fixed Length Unit Tests:** UTRL trains LLMs to generate a fixed number of test cases per unit test. Future work could explore adaptive generation of variable-length unit tests.
