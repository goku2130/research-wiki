---
id: arxiv:2305.20050
type: paper
title: Let's Verify Step by Step (Lightman et al., 2023)
url: https://arxiv.org/abs/2305.20050
retrieved: '2026-07-12'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

# Summary: Let’s Verify Step by Step (Lightman et al., 2023)

### Core Problem
Large language models (LLMs) frequently produce logical errors or "hallucinations" during complex multi-step reasoning. A single error in a chain-of-thought can derail an entire solution. To mitigate this, the authors investigate whether **process supervision** (providing feedback for each intermediate reasoning step) is more effective than **outcome supervision** (providing feedback only for the final result) when training reward models to identify correct solutions.

### Method
The researchers compared Outcome-supervised Reward Models (ORMs) and Process-supervised Reward Models (PRMs) using the challenging MATH dataset.

#### 1. Generator Setup
A generator model (based on GPT-4 for large-scale experiments) was used to produce solutions. To improve mathematical reasoning, all models were first finetuned on **MathMix**, a dataset of 1.5 billion math-relevant tokens. The generator was further finetuned to produce solutions in a newline-delimited, step-by-step format.

#### 2. Reward Model Training
*   **ORM:** Trained to predict whether a complete solution is correct or incorrect based on the final answer.
*   **PRM:** Trained to predict the correctness of each individual step. The authors released **PRM800K**, a dataset containing 800,000 step-level human labels (positive, negative, or neutral) across 75,000 solutions.
*   **Active Learning:** To increase data efficiency, the authors used a "convincing wrong-answer" strategy. They surfaced solutions that were rated highly by the current PRM but reached an incorrect final answer, forcing the model to learn from its most confident mistakes.

#### 3. Scoring and Evaluation
The models were evaluated using **best-of-N search**: the generator produces $N$ solutions, the reward model ranks them, and the top-ranked solution is graded. 

For the PRM, the overall solution score is calculated as the product of the correctness probabilities for every step:

$$
\text{Score}(\text{solution}) = \prod_{i=1}^{k} P(\text{step}_i \text{ is correct})
$$

(Where $k$ is the total number of steps in the solution).

### Key Quantitative Results
*   **Performance:** On a representative subset of the MATH test set (Best-of-1860), the PRM solved **78.2%** of problems, significantly outperforming the ORM (**72.4%**) and the majority voting baseline (**69.6%**).
*   **Data Efficiency:** Active learning provided an approximate **2.6$\times$ improvement** in data efficiency for process supervision compared to uniform sampling.
*   **OOD Generalization:** On held-out STEM tests (AP Physics, Calculus, Chemistry, AMC10/12), the PRM consistently outperformed the ORM. For example, in AP Calculus, the PRM achieved **86.7%** accuracy compared to the ORM's **68.9%** (Best-of-100).
*   **Scaling:** Small-scale synthetic experiments confirmed that process supervision outperforms outcome supervision across various data scales.

### Stated Limitations
*   **Test Set Contamination:** The authors acknowledge that some MATH dataset problems may have appeared in the pretraining data, potentially inflating results, though OOD tests suggest the relative gains of PRMs remain valid.
*   **False Positives:** PRMs can still be fooled by subtle errors, such as counting mistakes or "convincing" but incorrect reasoning that happens to lead to a correct final answer.
*   **Active Learning Instability:** Preliminary attempts to iteratively retrain the selector model during active learning resulted in instability and did not improve performance.
