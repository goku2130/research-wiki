---
id: arxiv:2308.01320
type: paper
title: 'DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models
  at All Scales'
url: https://arxiv.org/abs/2308.01320
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

# DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training

DeepSpeed-Chat is an end-to-end system designed to democratize the training of ChatGPT-like models by making Reinforcement Learning from Human Feedback (RLHF) accessible, efficient, and cost-effective.

### Core Problem
Training large-scale models (billions of parameters) via RLHF is traditionally prohibitively expensive and computationally inefficient. The authors note that existing systems often operate at less than 5% of the hardware's peak capability. The primary bottlenecks occur during the RLHF stage, which requires managing multiple model copies (actor, reward, reference, and critic), leading to massive memory overhead and slow token generation during the experience collection phase.

### Method and Pipeline
DeepSpeed-Chat implements a three-step training pipeline based on the InstructGPT framework:

1.  **Supervised Fine-Tuning (SFT):** Pre-trained language models are fine-tuned on carefully selected human responses to various queries.
2.  **Reward Model (RW) Fine-Tuning:** A separate, typically smaller model is trained using a dataset of human-provided rankings of multiple answers to the same query.
3.  **RLHF Training:** The SFT model is further optimized using the reward feedback from the RW model via the **Proximal Policy Optimization (PPO)** algorithm.

To enhance model quality, the system includes two optional features in Step 3:
*   **Exponential Moving Average (EMA):** Collection of EMA-based checkpoints to improve final response quality.
*   **Mixture Training:** Mixing the pre-training objective (next-word prediction) with the PPO objective to prevent performance regression on public benchmarks (e.g., SQuAD2.0).

#### DeepSpeed Hybrid Engine
The technical core of the system is the **Hybrid Engine**, which provides a unified infrastructure for both training and inference. It allows the system to seamlessly switch between:
*   **Inference Mode:** Used for experience generation. It employs a lightweight memory management system for KV-cache, optimized inference kernels, and tensor parallelism (TP) to maximize throughput.
*   **Training Mode:** Used for weight updates. It leverages the **ZeRO** (Zero Redundancy Optimizer) family of technologies and Low-Rank Adaptation (LoRA) to optimize memory sharding.

### Key Quantitative Results
DeepSpeed-Chat demonstrates significant improvements in throughput and scalability compared to frameworks like Colossal-AI and HuggingFace DDP.

**Training Efficiency and Cost:**
*   **OPT-13B:** Trained in 9 hours on Azure Cloud for under \$300.
*   **OPT-30B:** Trained in 18 hours for under \$600.
*   **OPT-175B:** Trained in under one day using a 64-GPU cluster.

**End-to-End (E2E) Time Breakdowns:**
| Model Size | Hardware | Step 1 (SFT) | Step 2 (RW) | Step 3 (RLHF) | Total Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OPT-1.3B** | 1 $\times$ A6000 (48GB) | 2900s | 670s | 1.2hr | 2.2hr |
| **OPT-13B** | 8 $\times$ A100-40G | 2.5hr | 0.25hr | 10.8hr | 13.6hr |
| **OPT-66B** | 64 $\times$ A100-80G | 82min | 5min | 7.5hr | 9hr |

**Comparative Performance:**
*   **Throughput:** Achieves over 10$\times$ improvement on a single GPU. On multi-GPU setups, it shows a 6–19$\times$ speedup over Colossal-AI and a 1.4–10.5$\times$ speedup over HuggingFace DDP.
*   **Model Scalability:** On a single A100-40G node, DeepSpeed-Chat can run a 50B model, whereas Colossal-AI is limited to 6.7B (a 7.5$\times$ increase in scale).

### Limitations
The authors note that for extremely large models, such as those with 175B parameters, effective throughput drops. This is attributed to limited memory, which restricts the maximum possible batch size. While still 1.2$\times$ more efficient than the 1.3B model per GPU, the system's efficiency for these "gigantic models" is constrained by the available memory per GPU.
