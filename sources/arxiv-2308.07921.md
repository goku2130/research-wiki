---
id: arxiv:2308.07921
type: paper
title: Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based
  Self-Verification
url: https://arxiv.org/abs/2308.07921
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

# Summary: Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification

### Core Problem
While Large Language Models (LLMs) have advanced in reasoning, they often struggle with complex mathematical word problems, frequently producing calculation errors or illogical reasoning steps. Although GPT-4 Code Interpreter (GPT4-Code) improves performance by generating and executing Python code, it primarily performs "self-debugging" (rectifying code execution errors) rather than verifying the logical correctness of the final answer or the overall reasoning chain.

### Method
The authors propose a framework to leverage GPT4-Code's inherent ability to execute code for autonomous verification and refinement. The method consists of two primary components:

**1. Explicit Code-based Self-Verification (CSV)**
CSV is a zero-shot prompting technique that guides the model to explicitly validate its own solution using code. The process follows a conditional loop:
*   **Step 1:** The model solves the problem using its standard code-generation and execution process.
*   **Step 2:** The model is prompted to generate additional code to verify the resulting answer.
*   **Step 3:** The verification state is classified as **True**, **False**, or **Uncertain**.
*   **Step 4:** If the state is **False**, the model automatically amends its reasoning steps and repeats the process until a **True** or **Uncertain** state is reached. If **True** or **Uncertain**, the answer is finalized.

**2. Verification-Guided Weighted Majority Voting (VW-Voting)**
To improve the reliability of the final output, the authors replace naive majority voting with a weighted system. They sample $k$ independent reasoning paths and assign weights to the final answers based on their self-verification state, where $w_{T} > w_{U} > w_{F}$ (True > Uncertain > False).

The voting score for a candidate answer $a$ is calculated as:

$$
\text{Score}(a) = \sum_{\{v^i\}} w_v (\# \{i \mid a^i = a \text{ and } v^i = v\}), \quad v \in \{\text{True, Uncertain, False}\}
$$

The final output is the answer with the highest score:

$$
\text{Output} = \underset{a}{\arg \max} \text{Score}(a)
$$

### Key Quantitative Results
The proposed method was evaluated on several benchmarks, showing significant gains over the base GPT4-Code model:

*   **MATH Dataset:**
    *   **GPT4-Code (Baseline):** 69.69% accuracy.
    *   **GPT4-Code + CSV:** 73.54% accuracy.
    *   **GPT4-Code + CSV + VW-Voting ($k=16$):** 84.32% accuracy.
*   **GSM8K Dataset:** Achieved 97.0% accuracy with $k=5$ sampled paths.
*   **MMLU-Math:** Achieved 89.2% accuracy (zero-shot).
*   **MMLU-STEM:** Achieved 87.0% accuracy (zero-shot).
*   **Verification Precision:** The average precision of the self-verification process was found to be 95.88%.

The authors also noted a strong positive correlation between **Code Usage Frequency** and accuracy, particularly for higher-difficulty problems.

### Limitations
The authors identify two primary limitations:
1.  **Model Specificity:** The analysis and improvements are currently focused exclusively on GPT4-Code; the generalizability to other LLMs remains to be explored.
2.  **Domain Constraints:** The method showed minimal improvement in **Geometry** problems (+0.6% accuracy), which the authors attribute to the necessity of multi-modal capabilities (visual reasoning) that are beyond the scope of a text-and-code-based approach.
