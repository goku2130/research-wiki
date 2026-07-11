---
id: interconnects:rlhf-roundup-trying-to-get-good-at-ppo-i
type: web
title: 'RLHF roundup: Trying to get good at PPO (Interconnects)'
url: https://www.interconnects.ai/p/rlhf-roundup-2024
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

# Summary: RLHF Roundup: Trying to Get Good at PPO

### Core Problem
The open-source alignment community has become heavily reliant on Direct Preference Optimization (DPO) because the engineering stack for Proximal Policy Optimization (PPO) is perceived as fickle and "broken" compared to the techniques used in industry. While industry reports suggest that PPO can outperform DPO by more than 5% on average, open-source practitioners have struggled to replicate these gains. The core objective of the research described is to determine if PPO can be made to work as effectively as industry claims and to disentangle the impact of algorithmic choice versus dataset quality.

### Method and Recipe
The researchers utilized a new PPO implementation written in Jax to test performance on a Llama 2 13B base model, evaluated using the `open-instruct` suite. The experimental process followed these steps:

1.  **Establish DPO Baseline (Noisy Data):** Trained a DPO model on the instruction-tuned variant using the Anthropic Helpful and Harmless (HH-RLHF) dataset.
2.  **Establish DPO Baseline (High-Quality Data):** Trained a DPO model using the UltraFeedback synthetic dataset to isolate the impact of data quality.
3.  **PPO Implementation:** Applied the Jax-based PPO implementation while keeping the dataset constant (UltraFeedback).
4.  **Iterative Refinement (Attempted):** The team tested two additional hypotheses to further extract performance:
    *   **Reward Model Scaling:** Using a 70B parameter reward model to distill knowledge into the 13B policy.
    *   **Reasoning Augmentation:** Adding more reasoning prompts to the training mix to improve performance on code and reasoning tasks.

### Key Quantitative Results
The study measured performance gains relative to the instruction-tuned baseline:
*   **DPO (HH-RLHF data):** Resulted in a **+1.3 score**.
*   **DPO (UltraFeedback data):** Resulted in a **+2.9 score**.
*   **PPO (UltraFeedback data):** Provided an additional **+1.2 score** over the DPO UltraFeedback baseline.

Other noted quantitative findings include:
*   **Post-training Impact:** Llama 3 Instruct improved MMLU scores by nearly **2.5 points** through post-training.
*   **RewardBench:** Peak accuracy for top reward models has increased from the **60-80%** range to over **90%**.
*   **LMSYS Competition:** The human preference prediction dataset consists of over **55,000** real-world conversations for training and **25,000** samples for testing.

*(Note: The source text does not provide specific mathematical formulas for PPO or DPO.)*

### Stated Limitations
Despite the gains, several limitations were identified:
*   **Model Behavior:** PPO-trained models remained "extremely yappy and a little unhinged," suggesting that high scores on AlpacaEval (length controlled) can be misleading.
*   **Generalizability:** Success with PPO was more consistent on Llama 2 base models; the researchers had less success transitioning PPO to other base models.
*   **Scale Gap:** Open-source 70B models remain significantly below the performance of Meta’s Llama-3-Instruct.
*   **Refinement Failure:** Neither the use of a larger (70B) reward model nor the addition of reasoning prompts yielded reliable, consistent performance increases across benchmarks.
*   **Evaluation Tools:** RewardBench is noted to have bugs in its dataset and uncertain correlation with downstream RLHF performance.
*   **Data Noise:** In the LMSYS competition, the extreme diversity of prompts introduces noise that likely lowers the maximum possible accuracy.
