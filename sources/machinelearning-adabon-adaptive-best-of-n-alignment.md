---
id: machinelearning:adabon-adaptive-best-of-n-alignment
type: web
title: 'AdaBoN: Adaptive Best-of-N Alignment'
url: https://machinelearning.apple.com/research/best-of-n
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# AdaBoN: Adaptive Best-of-N Alignment

### Core Problem
The primary challenge addressed by AdaBoN is the computational inefficiency of test-time alignment methods, specifically **Best-of-N sampling**. While Best-of-N sampling effectively steers language models (LMs) toward preferred behaviors by using reward models (RM) to select the best output from $N$ samples, it is typically applied uniformly across all prompts. This uniform approach is computationally expensive and fails to account for the fact that different prompts have varying levels of "alignment difficulty," meaning some prompts require more samples than others to find a high-reward response.

### Method
AdaBoN introduces a prompt-adaptive strategy designed to allocate inference-time compute more efficiently. The method is designed to be practical and compatible with any combination of LMs and RMs. The algorithm operates in two distinct stages:

1.  **Exploratory Phase**: The system utilizes a small, initial exploration budget to sample responses for each prompt. The goal of this phase is to estimate the reward distribution associated with each specific prompt.
2.  **Adaptive Allocation Phase**: Based on the reward distribution estimates obtained in the first stage, the algorithm adaptively allocates the remaining inference budget. Prompts that demonstrate a greater need for sampling (based on the estimated distribution) receive a larger portion of the remaining budget to optimize the final output selection.

### Key Formulas
The provided source text does not list specific mathematical formulas for the reward distribution estimation or the allocation logic.

### Quantitative Results
The effectiveness of AdaBoN was evaluated using 12 different LM/RM pairs across 50 different batches of prompts, utilizing three datasets: **AlpacaEval**, **HH-RLHF**, and **PKU-SafeRLHF**. The key findings include:

*   **Budget Efficiency**: AdaBoN outperforms uniform allocation when using the same total inference budget.
*   **Competitive Performance**: The adaptive strategy remains competitive even when compared to uniform allocation methods that are granted a **20 percent larger** inference budget.
*   **Scalability**: The performance of the adaptive strategy improves as the batch size of the prompts increases.

### Stated Limitations
The provided text does not explicitly state any limitations of the AdaBoN method.
