---
id: arxiv:2402.03300
type: paper
title: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models'
url: https://arxiv.org/abs/2402.03300
retrieved: '2026-07-11'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

**Core Problem**
Mathematical reasoning remains a significant challenge for large language models due to its complex, structured nature. While closed-source models like GPT-4 and Gemini-Ultra demonstrate strong capabilities, open-source models substantially lag behind. The core objective is to bridge this performance gap by leveraging publicly available web data and developing an efficient reinforcement learning algorithm tailored for mathematical reasoning.

**Method/Recipe Step by Step**
The training pipeline proceeds in four sequential stages:
1. **Iterative Data Collection:** A fastText-based classifier is trained using OpenWebMath as positive examples and Common Crawl pages as negatives. The model recalls mathematical web pages from deduplicated Common Crawl, ranking them by predicted scores. Over four iterations, domains with >10% math content are identified, URLs are manually annotated, and the classifier is retrained to capture diverse mathematical sources. Benchmark contamination is filtered using exact 10-gram matching, yielding 35.5M pages totaling 120B tokens.
2. **Continual Pre-training:** The base model initializes from DeepSeek-Coder-Base-v1.5 7B and trains for 500B tokens. The data distribution comprises 56% DeepSeekMath Corpus, 20% GitHub code, 10% natural language, 10% arXiv papers, and 4% AlgebraicStack.
3. **Supervised Fine-Tuning (SFT):** The model undergoes instruction tuning on 776K examples formatted as chain-of-thought, program-of-thought, and tool-integrated reasoning across English and Chinese benchmarks. Training runs for 500 steps with a batch size of 256, constant learning rate of 5e-5, and 4K context length.
4. **Reinforcement Learning (GRPO):** For each question, the policy samples $G$ outputs, computes rewards via a reward model, and normalizes them within the group. The policy is updated using outcome or process supervision. Iterative RL continuously retraining the reward model using a 10% historical replay buffer. The reference model is reset to the current policy after each iteration.

**Key Formulas**
GRPO modifies standard Proximal Policy Optimization by eliminating the critic model and estimating the baseline from group scores. The GRPO objective is:
\[
\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q) \right] \frac{1}{G} \sum_{i=1}^G
