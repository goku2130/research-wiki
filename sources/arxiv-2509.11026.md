---
id: arxiv:2509.11026
type: paper
title: Rethinking Human Preference Evaluation of LLM Rationales
url: https://arxiv.org/abs/2509.11026
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# Rethinking Human Preference Evaluation of LLM Rationales

## Core Problem
The evaluation of Large Language Model (LLM) generated rationales—free-form explanations used to improve reasoning and interpretability—currently relies heavily on binary human preference judgments (win/loss). The authors argue that this approach is opaque and coarse-grained, failing to provide insight into the specific attributes that make one rationale superior to another. Consequently, it remains unclear whether binary preferences truly reflect rationale quality or which specific dimensions of a rationale most influence human judgment.

## Method
The researchers proposed a framework to decompose "preference" into fine-grained attributes and analyze their impact using the following step-by-step recipe:

1.  **Attribute Identification**: The authors synthesized 12 key rationale attributes from prior literature: Faithfulness, Hallucination, Repetition, Informativeness, Plausibility, Self-Consistency, Source Consistency, Grammar, Arithmetic Accuracy, Conciseness, Completeness, and Correctness.
2.  **Attribute Measurement**: These attributes were quantified using three distinct methods:
    *   **Automated Heuristics**: Utilizing ROSCOE metrics for baseline scoring.
    *   **LLM Judges**: Employing GPT-4o and Gemini 2.5-Flash (0–1 scale) and OLMo 32B (0–10 scale).
    *   **Human Annotations**: Expert annotations performed by the three co-first authors (0–1 scale).
3.  **Preference Explanation (SHAP Analysis)**: To determine which attributes drive human preference, the authors used two datasets: **Chatbot Arena** (filtered to 1,367 math/logic questions) and **MT-Bench** (80 unique reasoning questions). They trained a **LightGBM** (gradient-boosted decision tree) model where attributes served as input features ($X$) and human preference labels served as the target variable ($y$). They then applied **SHAP (SHapley Additive exPlanations)** to quantify the contribution of each attribute to the prediction.
4.  **Fine-Grained Re-evaluation**: The authors computed attribute-specific ELO ratings. Instead of using binary preference labels, they used the mean scores from three LLM judges to determine the "winner" for each specific attribute, allowing for a multi-dimensional ranking of models.

## Key Quantitative Results
*   **Predictors of Preference**: SHAP analysis revealed that **Correctness**, **Plausibility**, and **Completeness** are the most influential attributes predicting human preference across datasets and judges. These were followed in importance by **Informativeness** and **Conciseness**.
*   **Model Comparisons**: While GPT-4, GPT-3.5-Turbo, and Claude-v1 consistently ranked in the top three, attribute-specific ELOs uncovered nuances obscured by binary scores:
    *   **Claude-v1** performed poorly regarding **Repetition** across both datasets.
    *   **GPT-3.5-Turbo** unexpectedly outperformed **GPT-4** in **Arithmetic Accuracy** and **Self-Consistency**.

## Limitations
The authors identify three primary limitations:
*   **LLM-as-a-Judge Reliability**: The study relies on LLMs for filtering and scoring, which introduces risks of non-deterministic outputs, inherent judge bias, and factual errors (e.g., GPT-4o assigning a perfect correctness score to a rationale containing an algebraic error).
*   **Task Scope**: The analysis was restricted to mathematical and logical reasoning; the importance of attributes may differ in creative writing or commonsense inference tasks.
*   **Human Annotation Scale**: Human annotations were conducted only by the three co-first authors, lacking a larger, more diverse pool of annotators and formal inter-annotator agreement (IAA) measurements.
