---
id: arxiv:2402.03300
type: paper
title: 'GRPO: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models'
url: https://arxiv.org/abs/2402.03300
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

DeepSeekMath addresses the performance gap between open-source and proprietary large language models (LLMs) in mathematical reasoning. The authors propose a comprehensive pipeline involving large-scale domain-specific pre-training, supervised fine-tuning (SFT), and a novel reinforcement learning (RL) algorithm to enhance reasoning capabilities.

### Method and Recipe

The development of DeepSeekMath 7B follows a three-stage process:

**1. Scalable Math Pre-training**
The model is initialized from **DeepSeek-Coder-Base-v1.5 7B**. The authors constructed the **DeepSeekMath Corpus**, containing 120B math-related tokens from Common Crawl, using an iterative data selection pipeline:
*   **Seed Selection:** Used OpenWebMath as the initial seed.
*   **Classification:** Trained a fastText classifier using seed data (positive) and random Common Crawl pages (negative).
*   **Iterative Refinement:** Mined positive instances, refined them via human annotation, and updated the classifier over four iterations.
*   **Training Mix:** The final pre-training distribution consisted of 56% DeepSeekMath Corpus, 20% GitHub code, 10% arXiv, 10% natural language (Common Crawl), and 4% AlgebraicStack.

**2. Supervised Fine-Tuning (SFT)**
DeepSeekMath-Base was fine-tuned on 776K examples. The dataset included English and Chinese problems paired with solutions in three formats: Chain-of-Thought (CoT), Program-of-Thought (PoT), and tool-integrated reasoning.

**3. Group Relative Policy Optimization (GRPO)**
To improve the SFT model without the heavy computational overhead of Proximal Policy Optimization (PPO), the authors introduced **GRPO**. Unlike PPO, which requires a separate critic (value) model to estimate the baseline, GRPO estimates the baseline from the average reward of a group of sampled outputs for the same question.

The GRPO objective is defined as:

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E} [ q \sim P(Q), \{o_{i} \}_{i = 1}^{G} \sim \pi_{\theta_{old}} (O | q) ] \frac{1}{G} \sum_{i = 1}^{G} \frac{1}{| o_{i} |} \sum_{t = 1}^{| o_{i} |} \left\{\min \left[ \frac{\pi_{\theta} (o_{i, t} | q , o_{i, < t})}{\pi_{\theta_{old}} (o_{i, t} | q , o_{i, < t})} \hat{A}_{i, t}, \text{clip} \left(\frac{\pi_{\theta} (o_{i, t} | q , o_{i, < t})}{\pi_{\theta_{old}} (o_{i, t} | q , o_{i, < t})}, 1 - \varepsilon , 1 + \varepsilon\right) \hat{A}_{i, t} \right] - \beta \mathbb{D}_{KL} [ \pi_{\theta} || \pi_{ref} ] \right\}
$$

The KL divergence is estimated using an unbiased estimator:

$$
\mathbb{D}_{KL} [ \pi_{\theta} || \pi_{ref} ] = \frac{\pi_{ref} (o_{i, t} | q , o_{i, < t})}{\pi_{\theta} (o_{i, t} | q , o_{i, < t})} - \log \frac{\pi_{ref} (o_{i, t} | q , o_{i, < t})}{\pi_{\theta} (o_{i, t} | q , o_{i, < t})} - 1
$$

**Advantage Estimation ($\hat{A}_{i,t}$):**
*   **Outcome Supervision:** The advantage for all tokens is the normalized reward of the output: $\hat{A}_{i,t} = \widetilde{r}_{i} = \frac{r_{i} - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$.
*   **Process Supervision:** The advantage is the sum of normalized rewards from the current token to the end of the output: $\hat{A}_{i,t} = \sum_{index(j) \geq t} \widetilde{r}_{i}^{index(j)}$.

### Key Quantitative Results

DeepSeekMath-RL 7B achieved state-of-the-art performance among open-source models:

| Model | GSM8K (Top1) | MATH (Top1) | MATH (Self-Consistency 64) |
| :--- | :---: | :---: | :---: |
| DeepSeekMath-Base 7B | 64.2% | 36.2% | - |
| DeepSeekMath-Instruct 7B | 82.9% | 46.8% | - |
| **DeepSeekMath-RL 7B** | **88.2%** | **51.7%** | **60.9%** |
| Minerva 540B (Base) | 58.8% | 33.6% | - |

**Other Findings:**
*   **Code Training:** Two-stage training (Code $\rightarrow$ Math) outperformed general pre-training followed by math training, suggesting code training enhances reasoning.
*   **arXiv Data:** The authors found that training exclusively on arXiv corpora (e.g., MathPile, ArXiv-RedPajama) provided no notable improvements or even led to deterioration in mathematical benchmarks.

### Stated Limitations

*   **Domain Weakness:** The model is relatively weaker in geometry and theorem-proving compared to closed-source models, specifically struggling with problems involving triangles and ellipses.
*   **Few-Shot Capability:** Due to model scale, DeepSeekMath shows similar performance in zero-shot and few-shot evaluations, whereas GPT-4 shows significant improvement with few-shot inputs.
