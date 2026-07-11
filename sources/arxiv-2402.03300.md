---
id: arxiv:2402.03300
type: paper
title: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models'
url: https://arxiv.org/abs/2402.03300
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

**Core Problem**
Mathematical reasoning remains a significant bottleneck for large language models, with open-source architectures lagging substantially behind proprietary systems like GPT-4 and Gemini-Ultra. The authors address this capability gap by developing DeepSeekMath, a 7B-parameter domain-specific model that maximizes mathematical reasoning through scalable web-data curation and a memory-efficient reinforcement learning algorithm.

**Methodology Step-by-Step**
The training pipeline executes four sequential stages. First, the DeepSeekMath Corpus is constructed via an iterative data selection pipeline from Common Crawl. Using OpenWebMath as a seed, a fastText classifier is trained with 500,000 positive and 500,000 negative samples to recall mathematical web pages. After URL-based deduplication, the top-ranked tokens are retained. Over four iterations, domains with high collection rates are manually annotated to enrich the seed corpus, ultimately yielding 120B tokens. Benchmark contamination is prevented by filtering any text containing exact 10-gram matches with evaluation datasets. Second, pre-training initializes from DeepSeek-Coder-Base-v1.5 7B and continues for 500B tokens, comprising 56% DeepSeekMath Corpus, 20% GitHub code, 10% arXiv papers, 4% AlgebraicStack, and 10% natural language data. Third, supervised fine-tuning (SFT) applies 776K instruction-tuning examples formatted with chain-of-thought, program-of-thought, and tool-integrated reasoning across English and Chinese benchmarks. Fourth, Group Relative Policy Optimization (GRPO) is applied to the SFT model. GRPO samples $G$ outputs per question, computes rewards via a reward model, and normalizes them by group statistics to estimate advantages without a separate critic model. The policy is updated using a single optimization step per exploration phase.

**Key Formulas**
The GRPO objective modifies the standard Proximal Policy Optimization (PPO) formulation by replacing the value-function baseline with group-relative advantages. The PPO surrogate objective is defined as:
$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E} [ q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q) ] \frac{1}{|o|} \sum_{t=1}^{|
