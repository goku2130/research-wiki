---
id: arxiv:2307.15043
type: paper
title: Sycophancy in Large Language Models
url: https://arxiv.org/abs/2307.15043
retrieved: '2026-07-11'
maturity: comprehensive
topic: sycophancy-and-misgeneralization
---

**Core Problem**
Aligned large language models (LLMs) are fine-tuned to refuse harmful or objectionable queries, yet manual "jailbreak" prompts remain brittle and require significant human engineering. Previous automatic adversarial prompt generation methods struggle with the discrete nature of token optimization and fail to generalize across diverse queries or architectures. This work addresses the vulnerability of post-hoc alignment by demonstrating that automated, gradient-based search can reliably discover universal adversarial suffixes that bypass safety guardrails across both open-source and proprietary models.

**Methodology**
The proposed Greedy Coordinate Gradient (GCG) attack constructs a single adversarial suffix appended to user queries through a structured three-step optimization process. First, the attack targets affirmative model responses rather than specific harmful outputs. The loss function forces the model to begin its completion with a compliant prefix (e.g., "Sure, here is...") followed by the original query, effectively switching the model into a generative mode. This objective is formalized by minimizing the negative log-likelihood of a target sequence $x_{n+1:n+H}^{\star}$:

$$
\mathcal{L}(x_{1:n}) = -\log p(x_{n+1:n+H}^{\star} | x_{1:n}) = -\log \prod_{i=1}^H p(x_{n+i} | x_{1:n+i-1}),
$$

where the optimization problem is $\underset{x_{\mathcal{I}} \in \{1, \dots, V\}^{|\mathcal{I}|}}{\text{minimize}} \mathcal{L}(x_{1:n})$. Second, discrete token optimization is performed via GCG. At each iteration, gradients with respect to one-hot token embeddings are computed: $\nabla_{e_{x_i}} \mathcal{L}(x_{1:n}) \in \mathbb{R}^{|V|}$. The top-$k$ tokens with the largest negative gradients are identified as candidates. A batch of candidates is evaluated via forward passes, and the substitution yielding the lowest loss is permanently applied. Third, to ensure universality, gradients and losses are aggregated across multiple harmful prompts and multiple models. Gradients are clipped to unit norm before aggregation. Prompts are added incrementally only after a candidate suffix succeeds on previously optimized queries, preventing early overfitting to a single instruction.

**Quantitative Results**
Evaluated on the AdvBench benchmark, GCG significantly outperforms baselines (PEZ, GBDA, AutoPrompt). Against Vicuna-7B, GCG achieves 88% Attack Success Rate (ASR) on individual harmful strings and 99% on individual harmful behaviors, with 100% train ASR and 98% test ASR across 25+ behaviors. Against LLaMA-2-7B-Chat, it achieves 57% ASR on strings and 84% test ASR on behaviors. Transfer attacks demonstrate remarkable cross-model efficacy. Suffixes optimized on Vicuna-7B/13B and Guanaco-7B/13B transfer to open-source models (Pythia, Falcon, ChatGLM) with near 100% ASR. Against proprietary black-box models, ensemble GCG prompts achieve 86.6% ASR on GPT-3.5, 46.9% on GPT-4, 47.9% on Claude-1, and 66% on PaLM-2. Claude-2 remains more robust at 2.1% ASR.

**Stated Limitations**
The authors identify several constraints. Optimization duration impacts transferability; running GCG for excessive steps (e.g., 500) causes overfitting to source models, reducing black-box success rates. Claude-2's higher robustness likely stems from pre-query content filters that block adversarial inputs before model evaluation, though these can potentially be evaded via multi-turn conditioning or prompt rewording. Discovered adversarial prompts exhibit variable interpretability; while some contain recognizable instructional phrasing, others resemble uninterpretable token sequences, suggesting multiple valid attack pathways exist. Finally, the study notes that current alignment efforts primarily defend against manual, natural-language attacks rather than automated gradient-based searches, suggesting that post-hoc alignment may be fundamentally insufficient without wholesale architectural or training paradigm shifts.
