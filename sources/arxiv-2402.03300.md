---
id: arxiv:2402.03300
type: paper
title: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models'
url: https://arxiv.org/abs/2402.03300
retrieved: '2026-07-11'
maturity: comprehensive
topic: grpo
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

DeepSeekMath addresses the performance gap between open-source language models and proprietary models (such as GPT-4 and Gemini-Ultra) in complex mathematical reasoning. The researchers propose a pipeline involving large-scale domain-specific pre-training, supervised fine-tuning (SFT), and a novel reinforcement learning (RL) algorithm to enhance reasoning capabilities.

### Methodology

#### 1. Math Pre-Training
The model, **DeepSeekMath-Base 7B**, is initialized from DeepSeek-Coder-Base-v1.5 7B. It undergoes continual training on 500B tokens, with a distribution consisting of 56% DeepSeekMath Corpus, 20% GitHub code, 10% arXiv, 10% natural language (Common Crawl), and 4% AlgebraicStack.

The **DeepSeekMath Corpus** (120B tokens) was constructed via an iterative pipeline:
1. **Seed Selection:** OpenWebMath was used as the initial seed.
2. **Classification:** A fastText-based classifier was trained using seed data as positive examples and random Common Crawl pages as negative examples.
3. **Mining:** The classifier recalled math-related pages from a deduplicated 40B HTML Common Crawl dataset.
4. **Iteration:** High-percentage math domains were identified, manually annotated, and added back to the seed corpus to refine the classifier over four iterations.

#### 2. Supervised Fine-Tuning (SFT)
**DeepSeekMath-Instruct 7B** was created by fine-tuning the base model on 776K examples. These included English and Chinese problems solved via Chain-of-Thought (CoT), Program-of-Thought (PoT), and tool-integrated reasoning formats.

#### 3. Group Relative Policy Optimization (GRPO)
To further improve reasoning, the authors introduced **GRPO**, a variant of Proximal Policy Optimization (PPO). Unlike PPO, which requires a separate critic (value) model to estimate the baseline—increasing memory and computational costs—GRPO estimates the baseline from the average reward of a group of sampled outputs for the same question.

The GRPO objective is defined as:

$$
\mathcal {J} _ {G R P O} (\theta) = \mathbb {E} [ q \sim P (Q), \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {o l d}} (O | q) ] \frac {1}{G} \sum_{i = 1} ^ {G} \frac {1}{| o _ {i} |} \sum_{t = 1} ^ {| o _ {i} |} \left\{\min \left[ \frac {\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta_ {o l d}} (o _ {i , t} | q , o _ {i , <   t})} \hat {A} _ {i, t}, \text{clip} \left(\frac {\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta_ {o l d}} (o _ {i , t} | q , o _ {i , <   t})}, 1 - \varepsilon , 1 + \varepsilon\right) \hat {A} _ {i, t} \right] - \beta \mathbb {D} _ {K L} [ \pi_ {\theta} | | \pi_ {r e f} ] \right\}
$$

Where $\hat{A}_{i,t}$ is the advantage calculated from relative rewards within the group. The KL divergence is estimated using an unbiased estimator:

$$
\mathbb {D} _ {K L} \left[ \pi_ {\theta} | | \pi_ {r e f} \right] = \frac {\pi_ {r e f} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})} - \log \frac {\pi_ {r e f} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})} - 1
$$

### Key Quantitative Results

DeepSeekMath-RL 7B achieved state-of-the-art performance for open-source models of its size:
*   **MATH Benchmark:** 51.7% Top1 accuracy (without external tools/voting). Using self-consistency over 64 samples, it reached 60.9%.
*   **GSM8K Benchmark:** 88.2% accuracy.
*   **Comparison:** DeepSeekMath-Base 7B (36.2% on MATH) outperformed the much larger Minerva 540B (33.6% on MATH).
*   **Out-of-Domain:** RL training on GSM8K and MATH improved performance on CMATH (84.6% $\rightarrow$ 88.8%).

### Findings and Limitations

**Key Findings:**
*   **Code Training:** Pre-training on code significantly benefits mathematical reasoning both with and without tool use.
*   **arXiv Data:** Contrary to common practice, the authors found that training on arXiv papers (e.g., MathPile, ArXiv-RedPajama) provided no notable improvements on the adopted benchmarks.
*   **RL Impact:** RL primarily enhances the model's output distribution (improving Maj@K) rather than increasing fundamental capabilities (Pass@K remained relatively stable).

**Limitations:**
*   **Domain Weakness:** The model is relatively weaker in geometry and theorem-proving, specifically struggling with problems involving triangles and ellipses.
*   **Few-Shot Capability:** Due to model scale, it lacks the strong few-shot improvement seen in GPT-4; its zero-shot and few-shot performances are similar.
