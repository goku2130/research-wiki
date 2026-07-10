---
id: arxiv:2402.03300
type: paper
title: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models'
url: https://arxiv.org/abs/2402.03300
retrieved: '2026-07-10'
maturity: comprehensive
topic: grpo
---

**Core Problem**
Mathematical reasoning remains a significant challenge for large language models due to its complex, structured nature. While closed-source models demonstrate strong capabilities, open-source models lag considerably behind. The primary objective is to bridge this performance gap by leveraging publicly available web data and developing efficient training paradigms without relying on external toolkits or voting techniques.

**Method and Training Recipe**
The DeepSeekMath framework follows a multi-stage pipeline. First, data curation begins with OpenWebMath as a seed corpus. A fastText classifier is trained using 500,000 positive examples and 500,000 negative examples from Common Crawl, configured with a vector dimension of 256, learning rate of 0.1, maximum word n-gram length of 3, minimum word occurrences of 3, and 3 training epochs. After URL and near-deduplication of Common Crawl (yielding 40B HTML pages), the classifier recalls mathematical pages. The top-ranked tokens are retained iteratively. To address seed diversity limitations, domains with >10% initial collection rates are identified, manually annotated for mathematical content, and added to the seed corpus for classifier retraining. This four-iteration process yields 35.5 million pages totaling 120B tokens. Benchmark contamination is removed via exact 10-gram matching against GSM8K, MATH, CMATH, and AGIEval. Second, continual pre-training initializes from DeepSeek-Coder-Base-v1.5 7B and undergoes 500B tokens of training comprising 56% DeepSeekMath Corpus, 4% AlgebraicStack, 10% arXiv, 20% GitHub code, and 10% multilingual natural language. Training employs AdamW with a multi-step learning rate schedule that peaks after 2,000 warmup steps, drops to 31.6% at 80% of training, and falls to 10.0% at 90%, using a peak learning rate of 4.2e-4, a batch size of 10M tokens, and a 4K context window. Third, supervised fine-tuning utilizes a 776K-example dataset formatted as chain-of-thought, program-of-thought, and tool-integrated reasoning. Fourth, reinforcement learning introduces Group Relative Policy Optimization (GRPO), a PPO variant that eliminates the critic model and estimates the advantage baseline directly from group scores, significantly reducing memory overhead.

**Key Formulas**
The source describes GRPO conceptually as a critic-free PPO variant that estimates baselines from group scores but does not provide explicit mathematical formulations for the policy optimization or baseline estimation steps. Consequently, no LaTeX expressions are extracted to strictly adhere to the source material.

**Key Quantitative Results**
DeepSeekMath-Base 7B achieves 64.2% on GSM8K and 36.2% on MATH, surpassing the 540B-parameter Minerva model. Following supervised fine-tuning, DeepSeekMath-Instruct 7B reaches 46.8% on MATH. GRPO reinforcement learning elevates MATH to 51.7% without external tools or voting, with self-consistency over 64 samples reaching 60.9%. In-domain gains include GSM8K (82.9% to 88.2%) and CMATH (84.6% to 88.8%). With Python tool use, the model achieves 66.9% on GSM8K and 31.4% on MATH. Formal autoformalization on miniF2F yields 25.8% (valid) and 24.6% (test). General capabilities improve to 54.9% on MMLU, 59.5% on BBH, 40.9% on HumanEval, and 52.6% on MBPP.

**Stated Limitations**
The methodology relies heavily on iterative manual annotation of web domains, introducing labor-intensive curation steps. The authors explicitly note that pre-training on arXiv papers yields no notable improvements across adopted mathematical benchmarks. Furthermore, the evaluation pipeline and corpus construction are initially optimized for English, requiring deliberate multilingual integration to mitigate performance degradation in Chinese benchmarks. The provided text excerpt concludes mid-sentence during the SFT data curation description, limiting the full specification of the instruction-tuning dataset composition.
