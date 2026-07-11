---
id: neurips:llm-evaluators-recognize-and-favor-their
type: web
title: LLM Evaluators Recognize and Favor Their Own Generations
url: https://neurips.cc/virtual/2024/poster/96672
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# LLM Evaluators Recognize and Favor Their Own Generations

### Core Problem
The use of Large Language Models (LLMs) as evaluators for their own outputs (self-evaluation) introduces systematic biases. A primary concern is **self-preference**, where an LLM scores its own generations higher than those of humans or other LLMs, even when human annotators perceive the quality as equal. This research investigates whether self-preference is driven by **self-recognition**—the model's empirical ability to distinguish its own writing style from that of others.

### Method
The researchers utilized 2,000 randomly sampled news articles from the **XSUM** and **CNN/DailyMail** datasets. They employed three instruction-tuned models: **GPT-4**, **GPT-3.5**, and **Llama-2-7b-chat**.

#### Measurement Recipe
The study measured self-recognition and self-preference using two distinct prompting formats:

1.  **Pairwise Measurement**: The LLM is presented with two texts (one self-generated, one from another source).
    *   **For Preference**: The model chooses the higher-quality text.
    *   **For Recognition**: The model identifies which text it generated.
    *   **Bias Mitigation**: To counter ordering bias, prompts are run twice with swapped options, and the results are averaged.
2.  **Individual Measurement**: The LLM is presented with a single text.
    *   **For Recognition**: A binary yes/no question regarding authorship.
    *   **For Preference**: A Likert-scale rating from 1 to 5.

#### Quantitative Metrics
Confidence and ratings were derived from the output probabilities of the target tokens:
*   **Pairwise Confidence**: Normalized output probabilities of the two options.

$$
\text{Confidence} = \frac{P(\text{option}_A)}{P(\text{option}_A) + P(\text{option}_B)}
$$

*   **Individual Likert Rating**: The weighted average of the five possible score tokens.

$$
\text{Rating} = \sum_{i=1}^{5} i \cdot P(\text{token}_i)
$$

#### Causal Analysis
To determine if self-recognition causes self-preference, the authors fine-tuned GPT-3.5 and Llama 2 on 500 examples to either increase or decrease their self-recognition capabilities. They also fine-tuned on a set of potential confounding properties to ensure the causal link was not due to other text characteristics.

### Key Quantitative Results
*   **Out-of-the-box Recognition**: All evaluated LLMs achieved over 50% accuracy in pairwise self-recognition. **GPT-4** demonstrated the highest capability, with **73.5% accuracy** in distinguishing its outputs from humans and two other LLMs.
*   **Fine-tuning Impact**: After fine-tuning on 500 examples, both GPT-3.5 and Llama 2 achieved **over 90% accuracy** in self-recognition.
*   **Correlation**: The researchers found a **linear correlation** between an LLM's self-recognition capability and the strength of its self-preference bias.
*   **Recognition Patterns**:
    *   Models most easily distinguished their own summaries from human-written ones.
    *   Llama 2 was unable to distinguish itself from GPT-3.5 or GPT-4.
    *   In individual settings, GPT-4 showed a strong bias *against* predicting any text as its own, regardless of the actual author.

### Limitations
The provided text notes that self-recognition can interfere with unbiased evaluations and AI safety. While the authors state that the causal explanation "resists straightforward confounders," the study focuses specifically on text summarization tasks, and the provided excerpt does not detail the specific nature of the confounding properties used in the control experiments.
