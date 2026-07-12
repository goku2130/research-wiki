---
id: arxiv:2212.10560
type: paper
title: 'Self-Instruct: Aligning Language Model with Self Generated Instructions'
url: https://arxiv.org/abs/2212.10560
retrieved: '2026-07-12'
maturity: comprehensive
topic: self-improvement-and-self-play
---

# Self-Instruct: Aligning Language Models with Self-Generated Instructions

### Core Problem
Instruction-tuned large language models (LMs) demonstrate strong zero-shot generalization, but their development depends heavily on human-written instruction data. This dependency creates a bottleneck because human-annotated data is costly to produce and often lacks the diversity and creativity required to cover the full variety of potential tasks, thereby limiting the generality of the resulting models.

### Method
SELF-INSTRUCT is a semi-automated bootstrapping framework that uses a pretrained LM to generate its own instruction-tuning data. The process begins with a small seed set of 175 manually written tasks (25 classification and 150 non-classification).

**The data generation pipeline follows four steps:**
1.  **Instruction Generation:** The LM is prompted with a few randomly sampled tasks from the task pool to generate new instructions for novel tasks.
2.  **Classification Task Identification:** The LM is prompted in a few-shot manner to determine if a generated instruction represents a classification task (defined as having a small, limited output label space).
3.  **Instance Generation:**
    *   **Input-first Approach:** For non-classification tasks, the LM generates the input fields first, followed by the corresponding output.
    *   **Output-first Approach:** For classification tasks, to avoid label bias (e.g., generating only grammatical inputs for error detection), the LM first generates possible class labels and then conditions the input generation on each label.
4.  **Filtering and Postprocessing:** To ensure diversity and quality, the framework:
    *   Removes instructions with a ROUGE-L similarity $\ge 0.7$ compared to any existing instruction in the pool.
    *   Excludes instructions containing keywords the LM cannot process (e.g., "image," "graph").
    *   Filters out identical instances or instances with the same input but different outputs.
    *   Applies heuristics to remove invalid generations (e.g., outputs that merely repeat the input).

**Finetuning:**
The original LM is finetuned using the generated data via supervised learning. The instruction and instance input are concatenated as a prompt. To increase robustness, the authors use multiple encoding templates (varying prefixes like "Task:" or "Input:" and different line-break configurations).

### Key Formulas
The framework defines instruction data as a set of instructions $\{I_t\}$, where each task $t$ has $n_t \ge 1$ input-output instances $\{(X_{t,i}, Y_{t,i})\}_{i=1}^{n_t}$. The goal is to train a model $M$ such that:

$$
M(I_t, X_{t,i}) = Y_{t,i}, \text{ for } i \in \{1, \dots, n_t\}
$$

### Quantitative Results
Applying SELF-INSTRUCT to vanilla GPT3 (davinci engine) produced a dataset of **52,445 instructions** and **82,439 instances**.

*   **Zero-Shot Generalization (SUPERNI):** $\text{GPT3}_{\text{SELF-INST}}$ achieved a **33.1% absolute improvement** over the vanilla GPT3 model, nearly matching the performance of InstructGPT001.
*   **Novel User-Oriented Tasks:** In a human evaluation of 252 expert-written instructions, $\text{GPT3}_{\text{SELF-INST}}$ outperformed models trained on public datasets (T0 and SUPERNI). It left only a **5% absolute gap** behind InstructGPT001 (when counting "Rating-B" acceptable responses as valid).
*   **Data Scaling:** Performance on user-oriented tasks improved as data size increased, plateauing after approximately **16K instructions**.
*   **Distillation:** Regenerating the output fields of the synthetic data using InstructGPT003 and then finetuning GPT3 resulted in a **10% performance increase** over the original $\text{GPT3}_{\text{SELF-INST}}$.

### Limitations
*   **Tail Phenomena:** The model may inherit the limitations of the base LM, where gains are skewed toward frequent language patterns in the pretraining corpus, potentially leading to brittleness with uncommon or highly creative instructions.
*   **Model Size Dependence:** The benefits of instruction-tuning are more pronounced in larger models.
*   **Bias Reinforcement:** The iterative process may amplify problematic social biases or struggle to produce balanced labels, reflecting the prior biases of the LM.
