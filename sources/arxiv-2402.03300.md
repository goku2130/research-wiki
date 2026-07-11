---
id: arxiv:2402.03300
type: paper
title: 'GRPO: Group Relative Policy Optimization'
url: https://arxiv.org/abs/2402.03300
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

**Core Problem**
Mathematical reasoning remains a significant bottleneck for large language models, with open-source architectures substantially trailing closed-source systems like GPT-4 and Gemini-Ultra. The authors identify two primary limitations in current pipelines: the systematic underutilization of publicly available web data for mathematical pre-training, and the prohibitive memory/computational overhead of reinforcement learning algorithms such as Proximal Policy Optimization (PPO), which require maintaining a separate critic/value model of comparable size to the policy network.

**Methodology**
The training recipe proceeds through four sequential stages. First, data collection employs an iterative fastText-based classifier initialized with OpenWebMath as positive examples and randomly sampled Common Crawl pages as negatives. After four refinement iterations incorporating human annotation, the pipeline yields the DeepSeekMath Corpus, comprising 120B tokens filtered for mathematical content and decontaminated via 10-gram exact matching against evaluation benchmarks. Second, pre-training initializes from DeepSeek-Coder-Base-v1.5 7B and continues for 500B tokens using a data mixture of 56% math corpus, 20% code, 10% natural language, 10% arXiv papers, and 4% AlgebraicStack. Third, supervised fine-tuning (SFT) applies 776K instruction-tuning examples featuring chain-of-thought, program-of-thought, and tool-integrated reasoning across English and Chinese datasets. Fourth, reinforcement learning
