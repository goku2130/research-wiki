---
id: github:a-survey-of-reinforcement-learning-for-l
type: web
title: A Survey of Reinforcement Learning for Large Reasoning Models
url: https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

# Summary: A Survey of Reinforcement Learning for Large Reasoning Models

This source is a comprehensive survey and curated repository maintained by TsinghuaC3I (Zhang et al., 2025) that examines the application of Reinforcement Learning (RL) to Large Reasoning Models (LRMs). The core objective of the work is to provide a structured taxonomy and a centralized knowledge base for the mechanisms, resources, and applications used to enhance the reasoning capabilities of large-scale models.

### Survey Taxonomy and Methodology
The survey organizes the field of RL for LRMs into five primary foundational pillars:

1.  **Foundational Components**: This section analyzes the technical "recipe" for building reasoning models, focusing on:
    *   **Reward Design**: Categorized into generative rewards, dense rewards, unsupervised rewards, and reward shaping.
    *   **Policy Optimization**: Examines various algorithmic approaches, including Policy Gradient objectives, critic-based algorithms, critic-free algorithms, off-policy optimization (including experience replay), and regularization objectives.
    *   **Sampling Strategy**: Investigates dynamic and structured sampling as well as the impact of sampling hyper-parameters.

2.  **Foundational Problems**: This section addresses the key debates and theoretical challenges inherent in applying RL to reasoning tasks.

3.  **Training Resources**: The survey catalogs the data and infrastructure required for training, divided into:
    *   **Static Corpora**: Specialized datasets covering Code, STEM, Mathematics, Agents, and mixed corpora.
    *   **Dynamic Environments**: Interactive training setups categorized as rule-based, code-based, game-based, model-based, and ensemble-based.
    *   **RL Infrastructure**: Primary and secondary tools used to implement these systems.

4.  **Applications**: The survey maps RL-enhanced reasoning to diverse real-world domains, including:
    *   **Agentic Systems**: Coding, search, browser-use, deep research, GUI/computer control, and recommendation agents.
    *   **Technical Tasks**: Code generation, software engineering, robotics, and scientific tasks.
    *   **Multimodal & Multi-Agent**: Multimodal understanding/generation and multi-agent systems.

5.  **Future Directions**: Identification of emerging research opportunities and unresolved challenges.

### Key Frameworks and Frontier Models
The source highlights several specialized frameworks developed by the authors/contributors:
*   **SSRL**: An investigation into Agentic Search RL that operates without reliance on external search engines.
*   **MARTI**: A framework designed for LLM-based Multi-Agent Reinforced Training and Inference.
*   **TTRL**: An open-source solution for online RL specifically for data lacking ground-truth labels, such as test data.

The survey also tracks "Frontier Models" in the reasoning space, including **Intern-S1**, **GLM-4.5** (Agentic, Reasoning, and Coding), **gpt-oss**, **InternVL3.5**, **Kimi K2**, **Step 3**, **GLM-4.1V-Thinking**, **Skywork-R1V3**, and **MiniMax-M1**.

### Quantitative Results and Formulas
The provided source text is a repository index and survey overview; it does not contain specific LaTeX formulas or quantitative performance metrics (such as accuracy percentages or benchmark scores) for the models listed.

### Stated Limitations
While the source does not list specific limitations of a single model, it explicitly identifies the existence of "Foundational Problems" and "challenges" as a core section of the survey, indicating that the integration of RL into LRMs remains a subject of active debate and technical difficulty.
