---
id: arxiv:2405.13432
type: paper
title: 'Disperse-Then-Merge: Pushing the Limits of Instruction Tuning via Alignment
  Tax Reduction'
url: https://arxiv.org/abs/2405.13432
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

# Disperse-Then-Merge (DTM): Reducing Alignment Tax in Instruction Tuning

### Core Problem
The authors address the "alignment tax," a phenomenon where Large Language Models (LLMs) experience a decline in performance on standard knowledge and reasoning benchmarks during the later stages of supervised fine-tuning (SFT) on instruction-following data. Through a pilot study, the authors hypothesize that this deterioration is not primarily caused by poor data quality or catastrophic forgetting of pre-training knowledge. Instead, they posit that as SFT progresses, the model begins to fit "data biases"—ungeneralizable, data-specific patterns—which outweigh the acquisition of generalizable instruction-following abilities and damage the model's parametric knowledge. This is evidenced by the ratio of training loss reduction to validation loss reduction ($\Delta\mathcal{L}_{train}/\Delta\mathcal{L}_{val}$), which increases from approximately $1.0$ at the start of training to nearly $20$ by the end.

### Method: Disperse-Then-Merge (DTM)
The DTM framework is designed to isolate and then eliminate these data-specific biases through a three-step process:

1.  **Instruction-following Data Distributing**: The instruction-following corpus $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ is partitioned into $K$ non-overlapping clusters $\{\mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_K\}$. This is achieved using K-means clustering based on cosine distance of sentence embeddings (obtained via models like MiniLM or MPNet) or through simple random splitting.
2.  **Sub-model Training**: The base LLM $\mathcal{M}_0$ is fine-tuned independently on each of the $K$ data portions. This results in a series of sub-models $\{\mathcal{M}_1, \mathcal{M}_2, \dots, \mathcal{M}_K\}$. Because each sub-model is trained on a smaller, different subset of data, the biases they acquire are distinct from one another.
3.  **Model Merging**: The sub-models are fused in the weight space to aggregate their generalizable instruction-following capacities while canceling out their unique, unshared data biases. The authors primarily utilize a weighted average of the sub-models:

$$
\mathcal{M}_{f}^{i}=\sum_{j=1}^{K}\alpha_{j}\mathcal{M}_{j}^{i}
$$

where $\mathcal{M}_{f}$ is the fused model, the superscript $i$ denotes a single parameter, and $\alpha_j$ is the merging weight for the $j$-th sub-model, subject to:

$$
\sum_{j=1}^{K}\alpha_{j}=1, \alpha_{j}\geq 0 \quad (j = 1, 2, \dots, K)
$$

### Key Quantitative Results
Experiments were conducted using Llama-2-7b as the backbone and the TÜLU-V2-mix dataset. With $K=4$ clusters and equal merging weights ($\alpha_j = 0.25$), DTM outperformed vanilla SFT and several baselines (including L2-norm, EWC, Replay, Uniform Soup, MoE, and Deita) across multiple benchmarks:

*   **Knowledge and Reasoning**: DTM achieved the highest or near-highest scores on several benchmarks, including **MMLU (50.43)**, **BBH (44.46)**, **GSM8K (20.62)**, and **TruthfulQA (29.13)**.
*   **Code Generation**: DTM reached **18.29 on HumanEval** and **23.60 on MBPP**.
*   **Instruction Following**: DTM improved upon vanilla SFT on **MT-bench (5.19 vs. 4.86)** and **Vicuna-bench (6.60 vs. 6.26)**.

The authors found that $K=4$ provided the best balance; too small a $K$ failed to sufficiently disperse biases, while too large a $K$ left sub-models with insufficient data to learn instruction-following. The framework also demonstrated robustness across other backbones, such as Mistral-7b and Baichuan-2-7b.

### Stated Limitations
The authors identify two primary limitations:
*   **Scope of Alignment**: The study focuses exclusively on SFT and does not investigate other alignment methods such as RLHF.
*   **Tuning Techniques**: The experiments primarily utilized LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning; the authors did not test other PEFT techniques (e.g., Adapter, IA3) or full-parameter fine-tuning.
