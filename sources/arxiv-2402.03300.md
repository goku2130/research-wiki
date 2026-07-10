---
id: arxiv:2402.03300
type: paper
title: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models'
url: https://arxiv.org/abs/2402.03300
retrieved: '2026-07-10'
maturity: comprehensive
topic: rl-for-reasoning
---

**Core Problem**
Mathematical reasoning remains a significant challenge for language models due to its highly structured and complex nature. While closed-source models like GPT-4 and Gemini-Ultra demonstrate strong capabilities, open-source models substantially trail in performance. The primary obstacles are the scarcity of high-quality, large-scale mathematical training data and the high computational overhead of reinforcement learning algorithms, which typically require extensive memory for critic models.

**Method/Recipe Step by Step**
The DeepSeekMath pipeline executes four sequential phases. First, data collection employs an iterative fastText-based classifier to mine mathematical content from Common Crawl. Starting with OpenWebMath as a seed, the classifier is trained on 500,000 positive and 500,000 negative examples. After URL-based deduplication, top-ranked pages are retained. The pipeline iterates four times, enriching the seed with manually annotated high-yield domains (e.g., mathoverflow.net) to improve recall, ultimately yielding 35.5 million pages totaling 120B tokens. Benchmark contamination is prevented by filtering any text containing 10-gram matches to evaluation datasets. Second, pre-training initializes from DeepSeek-Coder-Base-v1.5 7B and continues for 500B tokens using a mixed distribution: 56% DeepSeekMath Corpus, 20% GitHub code, 10% arXiv, 10% natural language, and 4% AlgebraicStack. Training utilizes AdamW with a multi-step learning rate schedule, a 10M token batch size, and a 4K context window. Third, supervised fine-tuning applies 776K instruction examples formatted with chain-of-thought, program-of-thought, and tool-integrated reasoning across English and Chinese. Finally, reinforcement learning employs Group Relative Policy Optimization (GRPO), a PPO variant that eliminates the critic model by estimating the baseline directly from group scores, trained on a subset of English instruction data.

**Key Formulas**
The provided source text describes GRPO conceptually as a critic-free PPO variant that uses group-score baselines to reduce memory overhead, but it does not provide explicit mathematical formulations, loss functions, or optimization equations for GRPO or the underlying reinforcement learning steps.

**Key Quantitative Results**
DeepSeekMath-Base 7B achieves 64.2% on GSM8K and 36.2% on MATH, surpassing the 540B-parameter Minerva model. Following instruction tuning, DeepSeekMath-Instruct 7B reaches 46.8% on MATH. GRPO reinforcement learning further elevates performance to 51.7% on MATH without external toolkits or voting, and 60.9% with self-consistency over 64 samples. In-domain gains include GSM8K improving from 82.9% to 88.2% and CMATH from 84.6% to 88.8%. With Python tool use, the model achieves 66.9% on GSM8K and 31.4% on MATH. General capabilities also improve, reaching 54.9% on MMLU, 59.5% on BBH, 40.9% on HumanEval, and 52.6% on MBPP.

**Stated Limitations**
The authors explicitly note that pre-training on arXiv papers yields no notable improvements across the adopted mathematical benchmarks. The data mining pipeline, while effective, relies on iterative web scraping and classification, which inherently depends on the quality and diversity of publicly available web pages. Additionally, the reinforcement learning phase described utilizes only a subset of English instruction-tuning data, leaving multilingual RL optimization unexplored in this iteration. The source text concludes mid-section during the SFT data curation description, indicating that further methodological details and comprehensive limitation discussions are truncated in the provided excerpt.
