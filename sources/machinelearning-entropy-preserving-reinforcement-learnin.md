---
id: machinelearning:entropy-preserving-reinforcement-learnin
type: web
title: Entropy-Preserving Reinforcement Learning
url: https://machinelearning.apple.com/research/entropy-preserving-reinforcement-learning
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

# Research Summary: Entropy-Preserving Reinforcement Learning

## Core Problem
The authors address a fundamental issue in policy gradient algorithms, particularly those used to enhance reasoning in large language models (LLMs). While these algorithms are valued for their ability to learn from exploration via their own trajectories—a process essential for generating diverse and creative solutions—they exhibit a natural tendency to reduce entropy during the training process. This reduction in entropy leads to a decrease in the diversity of explored trajectories, which ultimately limits the policy's capacity for further exploration and can hinder overall performance.

## Method
The researchers argue that entropy must be actively monitored and controlled throughout the training cycle. Their approach involves a three-part methodology:

1.  **Formal Analysis:** The authors conduct a formal analysis of how leading policy gradient objectives contribute to entropy dynamics.
2.  **Empirical Investigation:** They identify specific empirical factors that significantly influence entropy behavior, specifically highlighting the role of numerical precision.
3.  **Implementation of Control Mechanisms:** To counteract entropy decay, the authors propose two explicit mechanisms:
    *   **REPO:** A family of algorithms designed to regulate entropy by modifying the advantage function.
    *   **ADAPO:** An approach utilizing adaptive asymmetric clipping to maintain entropy.

## Key Formulas
The provided source text does not contain the specific mathematical formulas for the REPO or ADAPO algorithms.

## Quantitative Results
The source does not provide specific numerical data or benchmark scores. However, it states the following qualitative and comparative outcomes:
*   **Diversity:** Models trained with entropy-preserving methods (REPO and ADAPO) maintain trajectory diversity throughout the training process.
*   **Performance:** The resulting final policies are described as being "more performant" than those trained with standard policy gradient algorithms.
*   **Sequential Learning:** These methods allow policies to retain their "trainability" when introduced to sequential learning tasks in new environments.

## Stated Limitations
The provided text does not explicitly list any limitations of the proposed methods.
