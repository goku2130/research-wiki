---
id: arxiv:2505.13697
type: paper
title: RL in Name Only? Analyzing the Structural Assumptions in RL post-training for
  LLMs
url: https://arxiv.org/html/2505.13697
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

This work critically examines the structural assumptions underlying reinforcement learning (RL) post-training for large language models (LLMs), specifically analyzing Group Relative Policy Optimization (GRPO). The authors demonstrate that the standard Markov Decision Process (MDP) formulation used in LLM training—where states are deterministic token sequences, transitions are fixed concatenations, and rewards are assigned only at termination via an external verifier—degenerates the sequential decision process into a contextual bandit. Under this framework, RL updates naturally collapse into an on-policy variant of filtered iterative supervised fine-tuning (F-ISFT). Furthermore, the paper challenges the narrative that extended response lengths signify enhanced reasoning, proving instead that they are a direct artifact of uniformly distributing terminal rewards across all tokens.

The methodological recipe proceeds through the following sequential steps: (1) formalize LLM generation as an MDP under two critical structural assumptions: states are merely concatenated action histories, and terminal rewards or advantages are split uniformly across the entire trajectory; (2) simplify the standard GRPO objective by relaxing the KL-divergence penalty and assuming importance sampling ratios remain within clipping bounds; (3) decompose the simplified objective into positive ($\mathcal{G}^+$) and negative ($\mathcal{G}^-$) response groups based on binary verifier feedback; (4) derive the gradient update, which becomes:
\[
\nabla_{\theta} \mathcal{J}(\theta) = \mathbb{E} \left[ \frac{1}{G} \left[ A^+ \sum_{i=1}^{\mathcal{G}^+} \sum_{t=1}^{H} \mathcal{ISR}_{i,t}(\theta) \nabla_{\theta} \log \pi_{\theta}(o_{i,t}|q, o_{i,<t}) + A^- \sum_{i=1}^{\mathcal{G}^-} \sum_{t=1}^{H} \mathcal{ISR}_{i,t}(\theta) \nabla_{\theta} \log \pi_{\theta}(o_{i,t}|q, o_{i,<t}) \right] \right]
\]
where $\mathcal{ISR}$ denotes the importance sampling ratio, $A^+$ and $A^-$ are adaptive weights derived from group advantages, and $H$ is the sequence length; (5) empirically validate the equivalence by training models using GRPO and F-ISFT variants across multiple datasets and architectures.

Empirical validation was conducted on GSM8K, Countdown, and MATH-500 using Qwen-2.5 (0.5B, 1.5B), Llama-3.2 (1B, 3B-Instruct), Deepseek-math-7B-Instruct, and Qwen3-8B. Training utilized a batch size of 64, 5 samples per prompt, temperature 0.6, learning rate $10^{-6}$, and maximum response length 1024. Results confirm performance parity between GRPO and F-ISFT variants. For Qwen-2.5-0.5B on GSM8K, GRPO achieved 55.87% accuracy, while F-ISFT+- reached 55.72%. On Countdown, GRPO scored 37.73% versus F-ISFT+- at 37.86%. Larger models showed similar alignment: Deepseek-math-7B-Instruct achieved 82.4% (GRPO) and 83.7% (F-ISFT+-) on GSM8K, while Qwen3-8B reached 92.1% (GRPO) and 91.5% (F-ISFT+-). Llama-3.2-3B-Instruct attained 84.59% (GRPO) and 83.54% (F-ISFT+-) on GSM8K. Length analysis revealed that GRPO and F-ISFT+- both exhibit response elongation, driven primarily by increased lengths in incorrect responses due to uniform negative advantage distribution, contradicting claims that longer traces inherently improve reasoning.

The authors acknowledge several limitations. The theoretical equivalence strictly depends on the degenerate MDP assumptions; alternative formulations or reward structures may preserve genuine RL dynamics. Additionally, F-ISFT trained exclusively on negative samples (F-ISFT-) significantly underperforms on complex tasks like Countdown, indicating that positive supervision remains essential for robust generalization. The analysis also focuses narrowly on GRPO’s structural simplifications, leaving broader RL variants unexamined. Finally, while the paper identifies uniform credit assignment as the root cause of length bias, it notes that current mitigation strategies (e.g., length penalties) address superficial symptoms rather than the underlying formulation.
