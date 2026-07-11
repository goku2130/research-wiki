---
id: arxiv:2407.21530
type: paper
title: Data Contamination Report from the 2024 CONDA Shared Task
url: https://arxiv.org/abs/2407.21530
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# Summary: Data Contamination Report from the 2024 CONDA Shared Task

## Core Problem
Data contamination occurs in natural language processing when evaluation data is inadvertently included in the pre-training corpora of large-scale language models (LMs). This inclusion introduces biases and artificially inflates model performance on specific benchmarks, preventing a fair assessment of a model's true generalization capabilities. The scale of modern web-crawled data makes preventing and detecting this contamination exceptionally difficult.

## Method
The 1st Workshop on Data Contamination (CONDA 2024) implemented a shared task to create a structured, centralized public database of contamination evidence. The process followed these steps:

1.  **Evidence Collection:** Researchers submitted evidence of contamination via GitHub pull requests.
2.  **Reporting Requirements:** Contributors provided the contaminated resource (model or corpus), the affected evaluation dataset, a percentage breakdown of contamination for each split (train, development, and test), and a reference to the methodology or original scientific paper.
3.  **Categorization of Detection Methods:** The task classified evidence based on two primary detection approaches:
    *   **Data-based approaches:** These involve inspecting pre-training corpora using string or sub-string matching techniques. Examples include 13-gram overlap, 50-character overlap, or full-string overlap. These are further divided into *Proprietary data* (typically found in technical reports) and *Open data* (found via corpus analysis tools).
    *   **Model-based approaches:** These estimate contamination without accessing pre-training data, often formulated as Membership Inference Attacks (MIA). Techniques range from prompting models to generate verbatim evaluation data to analyzing output probabilities. These are categorized by their applicability to *Closed* or *Open* models.

## Key Quantitative Results
The report compiles 566 entries from 23 contributors, covering 91 datasets and 42 contaminated sources.

### General Statistics
*   **Contamination Events (>0%):** 432 total (317 test-set, 95 dev-set, and 20 train-set).
*   **Non-contamination Events (0%):** 144.

### Contaminated Corpora
The most frequently reported contaminated corpora include:
*   **C4:** 35 events
*   **RedPajama v2:** 32 events
*   **The Pile:** 30 events
*   **OSCAR:** 29 events
*   **CommonCrawl:** 6 events
*   **TheStack:** 2 events
*   **ProofPile:** 2 events
*   **xP3:** 1 event

### Contaminated Models
Evidence was reported for both closed and open models:
*   **Closed Models:** GPT-3 (24 events), GLaM (17), GPT-4 (16), GPT-3.5 (13), PaLM (8), PaLM-2 (3), GPT-3.5 Turbo (2), and Claude 3 Opus (1).
*   **Open Models:** FLAN-tuned models (14 events), Mistral (5), Llama 2 (3), Qwen (2), Llema (2), Aquila 2 (2), mT0 (1), and Bloom-Z (1).

### Task and Temporal Analysis
*   **Most Contaminated Tasks:** Text-scoring, Question Answering (QA), and multiple-choice-qa.
*   **Temporal Correlation:** Contamination aligns with release dates. GPT-3 (launched 2020) is predominantly contaminated with datasets from 2016, while GPT-4 (released 2023) is mainly contaminated with datasets from 2018 to 2022.

## Limitations
The authors state that the report does not cover all possible cases of data contamination due to the vast exploration space. The results represent a small sample of evidence reported specifically during the shared task period in mid-2024.
