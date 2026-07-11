---
id: arxiv:2402.03300
type: paper
title: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models (RLVR/GRPO)'
url: https://arxiv.org/abs/2402.03300
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

DeepSeekMath addresses the performance gap between open-source language models and proprietary models (such as GPT-4 and Gemini-Ultra) in complex mathematical reasoning. The authors propose a comprehensive pipeline involving large-scale domain-specific pre-training, supervised fine-tuning (SFT), and a novel reinforcement learning (RL) algorithm to enhance reasoning capabilities.

### Method and Recipe

The development of DeepSeekMath follows a three-stage process:

**1. Scalable Math Pre-training**
The model is initialized from **DeepSeek-Coder-Base-v1.5 7B**. The authors argue that starting from a code-centric model improves mathematical reasoning. They constructed the **DeepSeekMath Corpus** (120B tokens) using an iterative pipeline:
*   **Seed Selection:** Used OpenWebMath as the initial seed.
*   **Classification:** Trained a fastText-based classifier to identify math-related pages in Common Crawl.
*   **Iterative Refinement:** Identified math-related domains, performed human annotation on uncollected URLs, and updated the classifier over four iterations.
*   **Training Mix:** The final base model was trained on 500B tokens: 56% DeepSeekMath Corpus, 20% GitHub code, 10% arXiv, 10% natural language, and 4% AlgebraicStack.

**2. Supervised Fine-Tuning (SFT)**
DeepSeekMath-Base was fine-tuned on 776K examples using chain-of-thought (CoT), program-of-thought (PoT), and tool-integrated reasoning formats. This resulted in **DeepSeekMath-Instruct 7B**.

**3. Group Relative Policy Optimization (GRPO)**
To further improve performance, the authors introduced **GRPO**, a variant of Proximal Policy Optimization (PPO). Unlike PPO, which requires a separate critic (value) model to estimate the baseline—increasing memory and computational overhead—GRPO estimates the baseline from the average reward of a group of sampled outputs for the same question.

The GRPO objective is defined as:

$$
\mathcal {J} _ {G R P O} (\theta) = \mathbb {E} [ q \sim P (Q), \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {o l d}} (O | q) ] \frac {1}{G} \sum_{i = 1} ^ {G} \frac {1}{| o _ {i} |} \sum_{t = 1} ^ {| o _ {i} |} \left\{\min \left[ \frac {\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta_ {o l d}} (o _ {i , t} | q , o _ {i , <   t})} \hat {A} _ {i, t}, \text{clip} \left(\frac {\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta_ {o l d}} (o _ {i , t} | q , o _ {i , <   t})}, 1 - \varepsilon , 1 + \varepsilon\right) \hat {A} _ {i, t} \right] - \beta \mathbb {D} _ {K L} [ \pi_ {\theta} | | \pi_ {r e f} ] \right\}
$$

The advantage $\hat{A}_{i,t}$ is calculated based on relative rewards within the group. For outcome supervision, the normalized reward $\widetilde{r}_{i} = \frac{r_{i} - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$ is applied to all tokens. For process supervision, the advantage is the sum of normalized rewards from subsequent reasoning steps.

The KL divergence is estimated using an unbiased estimator:

$$
\mathbb {D} _ {K L} \left[ \pi_ {\theta} | | \pi_ {r e f} \right] = \frac {\pi_ {r e f} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})} - \log \frac {\pi_ {r e f} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})} - 1
$$

### Key Quantitative Results

*   **DeepSeekMath-RL 7B:** Achieved **51.7%** on the MATH benchmark (Top1) and **88.2%** on GSM8K. With self-consistency over 64 samples, it reached **60.9%** on MATH.
*   **Base Model Performance:** DeepSeekMath-Base 7B achieved **36.2%** on MATH and **64.2%** on GSM8K, outperforming the much larger Minerva 540B (33.6% on MATH).
*   **RL Gains:** GRPO improved MATH scores from 46.8% (Instruct) to 51.7% (RL) and GSM8K from 82.9% to 88.2%.
*   **Data Findings:** The authors found that code training significantly benefits both tool-use and non-tool-use math reasoning, whereas training on arXiv papers showed no notable improvement across the adopted benchmarks.

### Stated Limitations

*   **Domain Weaknesses:** The model performs relatively poorly on geometry and theorem-proving tasks (e.g., problems involving triangles and ellipses), suggesting data selection bias.
*   **Few-Shot Capability:** Due to its smaller scale compared to GPT-4, DeepSeekMath shows similar performance in zero-shot and few-shot evaluations, whereas GPT-4 improves significantly with few-shot inputs.
