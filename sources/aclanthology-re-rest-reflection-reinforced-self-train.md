---
id: aclanthology:re-rest-reflection-reinforced-self-train
type: web
title: 'Re-ReST: Reflection-Reinforced Self-Training for Language Agents'
url: https://aclanthology.org/2024.emnlp-main.861/
retrieved: '2026-07-12'
maturity: comprehensive
topic: self-improvement-and-self-play
---

# Re-ReST: Reflection-Reinforced Self-Training for Language Agents

## Core Problem
Finetuning language agents using reasoning-action trajectories is a highly effective method for improving performance. However, acquiring these trajectories is often impractical or prohibitively expensive when relying on human annotations or demonstrations from stronger models. While self-training—where the agent generates its own supervision—presents a viable alternative, it is fundamentally limited by the quality of the model-generated samples. In challenging language agent tasks, the agent often produces low-quality samples, which hinders the effectiveness of the self-training process.

## Method: Reflection-Reinforced Self-Training (Re-ReST)
Re-ReST addresses the quality gap in self-training by introducing a "reflector" mechanism to refine inferior samples. The process follows these steps:

1.  **Sample Generation**: The language agent generates initial reasoning-action trajectories for the target tasks.
2.  **Environmental Feedback**: The agent's outputs are submitted to an external environment to receive feedback. For example, in code generation tasks, this feedback consists of unit test results.
3.  **Refinement via Reflector**: A reflector takes two inputs—the agent's original output and the feedback from the external environment—to produce an improved, higher-quality version of the sample.
4.  **Dataset Enrichment**: These refined, high-quality samples are used to enrich the self-training dataset, replacing or augmenting the low-quality initial attempts.
5.  **Self-Training (Finetuning)**: The agent is finetuned on this enriched dataset of high-quality trajectories.
6.  **Inference Application**: To overcome the limitations of previous reflection methods, Re-ReST employs a method to utilize reflection during the inference phase without requiring ground-truth feedback.

## Key Formulas
The provided source text does not contain explicit mathematical formulas.

## Quantitative Results
The effectiveness of Re-ReST was evaluated across several tasks, including sequential decision-making, multi-hop question answering, visual question answering (VQA), text-to-image generation, and code generation. 

Key performance gains over baselines include:
*   **HotpotQA**: Standard self-training improved the baseline by **7.6%**, and the addition of Re-ReST provided a further boost of **2.0%**.
*   **AlfWorld**: Standard self-training improved the baseline by **28.4%**, and Re-ReST further increased performance by **14.1%**.

The results demonstrate that the reflector efficiently generates high-quality samples that significantly enhance the self-training process compared to standard self-training alone.

## Limitations
The authors note that previous reflection-based work was limited by the requirement for ground-truth feedback during the inference stage. Re-ReST specifically aims to address and resolve this limitation by demonstrating a method to employ reflection during inference without such feedback. No other specific limitations of the Re-ReST method itself are detailed in the provided text.
