---
id: github:a-survey-of-reinforcement-learning-for-l
type: web
title: A Survey of Reinforcement Learning for Large Reasoning Models
url: https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

This document, "A Survey of Reinforcement Learning for Large Reasoning Models," is a survey paper that comprehensively examines the integration of Reinforcement Learning (RL) with Large Reasoning Models (LRMs).

**Core Problem:**
The core problem addressed is the need for a structured understanding of how Reinforcement Learning (RL) can be effectively applied to enhance the capabilities of Large Reasoning Models (LRMs). This involves categorizing existing research, identifying foundational components and challenges, and outlining future research directions.

**Method/Recipe Step by Step:**
The survey organizes the field into five main sections to provide a structured overview:
1.  **Foundational Components:** This section delves into the core mechanisms of RL for LRMs, specifically focusing on:
    *   **Reward Design:** How rewards are formulated to guide LRM behavior. This includes sub-categories like Generative Rewards, Dense Rewards, Unsupervised Rewards, and Rewards Shaping.
    *   **Policy Optimization:** The algorithms and objectives used to update the LRM's policy. Sub-categories include Policy Gradient Objective, Critic-based Algorithms, Critic-Free Algorithms, Off-policy Optimization (with and without experience replay), and Regularization Objectives.
    *   **Sampling Strategy:** Methods for generating and selecting data during the RL process. This covers Dynamic and Structured Sampling, and Sampling Hyper-Parameters.
2.  **Foundational Problems:** This section discusses key debates and challenges inherent in combining RL with LRMs. (Details not provided in the excerpt).
3.  **Training Resources:** This section categorizes the data and environments used for training RL-enhanced LRMs, including:
    *   **Static Corpus:** Data types such as Code, STEM, Math, Agent-specific, and Mixed corpora.
    *   **Dynamic Environment:** Interactive environments categorized by their nature, such as Rule-based, Code-based, Game-based, Model-based, and Ensemble-based environments.
    *   **RL Infrastructure:** The primary and secondary tools and platforms supporting RL for LRMs.
4.  **Applications:** This section showcases real-world implementations of RL for LRMs across various domains, including:
    *   **Agent-based Applications:** Coding Agent, Search Agent, Browser-Use Agent, DeepResearch Agent, GUI&Computer Agent, Recommendation Agent, and other general Agents.
    *   **Task-specific Applications:** Code Generation, Software Engineering, Multimodal Understanding, Multimodal Generation, Robotics Tasks, Multi-Agent Systems, and Scientific Tasks.
5.  **Future Directions:** This section identifies emerging research opportunities and challenges in the field. (Details not provided in the excerpt).

**Key Formulas in LaTeX:**
The provided text does not contain any specific mathematical formulas in LaTeX.

**Key Quantitative Results and Numbers:**
The document does not present specific quantitative results or numbers from research studies. It lists various models and their release dates (e.g., Intern-S1 in 2025-08, GLM-4.5 in 2025-08, gpt-oss in 2025-08, InternVL3.5 in 2025-08, Kimi K2 in 2025-07, Step 3 in 2025-07, GLM-4.1V-Thinking in 2025-07, Skywork-R1V3 in 2025-07, GLM-4.5V in 2025-07, Magistral in 2025-06, Minimax-M1 in 2025-06, MiMo in 2025-05). It also notes that the survey was ranked "#1 Paper of the Day" on Hugging Face Daily Papers on 2025-09-12.

**Stated Limitations:**
The document explicitly states, "We welcome everyone to open an issue for any related work we haven’t discussed, and we’ll try to address it in the next release!" This indicates a recognition that the current survey might not be exhaustive and will be updated with new research.
