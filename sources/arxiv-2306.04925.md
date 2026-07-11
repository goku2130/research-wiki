---
id: arxiv:2306.04925
type: paper
title: 'DeepSpeed-Chat: Easy, Effective and Scalable Reinforcement Learning for Large
  Language Models'
url: https://arxiv.org/abs/2306.04925
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

**Core Problem**
The development of large-scale human-annotated NLP benchmarks is increasingly bottlenecked by the high cost and marginal accuracy gains of collecting new input-output pairs. The authors identify that annotating existing benchmarks with complementary signals is a more efficient alternative. Specifically, they address the limitation of instance-wise task labels by introducing task-specific pairwise preferences as an auxiliary annotation. This relative ordering captures fine-grained suitability information that standard labels miss, aiming to improve classifier accuracy, calibration, and robustness without the prohibitive cost of new data collection.

**Method/Recipe Step by Step**
The proposed Prefer-to-Classify (P2C) framework operates through a structured multi-task learning pipeline: (1) Acquire pairwise preference labels $y_{pref} \in \{0, 1, 0.5\}$ for input pairs $(\mathbf{x}^0, \mathbf{x}^1)$ using one of three sources: generative (querying LLMs like GPT-3), extractive (deriving signals from existing annotator vote counts), or subjective (crowd-sourced human judgments); (2) Initialize a Transformer backbone $g_\phi$ shared between a classification head $W_{task}$ and $T$ randomly initialized preference heads $W_{pref}^{(t)}$; (3) Train the preference predictors to output relative suitability probabilities while applying diversity regularization to prevent the multiple preference heads from collapsing into identical functions; (4) Enforce consistency regularization that explicitly constrains the classifier to assign higher confidence to preferred samples; (5) Implement disagreement-based or inconsistency-based sampling to select informative pairs, mitigating the quadratic scaling of possible comparisons; and (6) Jointly optimize the combined multi-task objective across all components.

**Key Formulas**
The preference predictor models the probability of $\mathbf{x}^1$ being preferred over $\mathbf{x}^0$ as:
\[
P_\psi[\mathbf{x}^1 \succ \mathbf{x}^0; y_{task}] = \frac{\exp(h_\psi(\mathbf{x}^1, y_{task}))}{\sum_{i \in \{0,1\}} \exp(h_\psi(\mathbf{x}^i, y_{task}))}
\]
The preference loss is formulated as binary cross-entropy:
\[
\mathcal{L}_{pref} = -\mathbb{E}\left[y_{pref}\log P_\psi[\mathbf{x}^1 \succ \mathbf{x}^0] + (1 - y_{pref})\log P_\psi[\mathbf{x}^0 \succ \mathbf{x}^1]\right]
\]
Diversity regularization maximizes KL-divergence across preference heads to maintain distinct learning signals:
\[
\mathcal{L}_{div} = \frac{-1}{T - 1} \sum_{j \neq i} D_{KL}\left(P_{\psi^{(i)}} \parallel P_{\psi^{(j)}}\right)
\]
Consistency regularization imposes a margin $m$ on classification confidence differences $\Delta p_y(\mathbf{x}^1, \mathbf{x}^0) = p_y(\mathbf{x}^0) - p_y(\mathbf{x}^1)$:
\[
\mathcal{L}_{cons} = y_{pref}\max\{0, m - \Delta p_y(\mathbf{x}^1, \mathbf{x}^0)\} + (1 - y_{pref})\max\{0, \Delta p_y(\mathbf{x}^1, \mathbf{x}^0) - m\}
\]
The total training objective aggregates these components: $\mathcal{L}_{train} = \mathcal{L}_{multi} + \lambda_{cons}\mathcal{L}_{cons}$.

**Key Quantitative Results**
Evaluated across ten text classification datasets and one image dataset, P2C consistently outperforms baselines. With GPT-3 generative preferences, P2C achieved an 11.55% relative test error reduction on average across four imbalanced datasets. Using extractive preferences from annotation records, P2C reduced test error by 7.59% versus vanilla training and 4.27% versus state-of-the-art disagreement baselines. Subjective preferences (5,000 pairs) yielded the highest accuracy and calibration, achieving an Expected Calibration Error of 6.09% compared to 9.19% for an equivalent number of task labels, with particularly strong gains on hard samples. In cross-domain validation on the SUN Attribute image dataset, P2C achieved a 6.66% relative test error reduction.

**Stated Limitations**
The framework's practical deployment is constrained by the cost and availability of preference signals. Subjective preference collection requires significant human labor, while generative preferences incur API costs. Extractive preferences are entirely dependent on the prior existence of detailed annotation records. The quadratic scaling of possible pairs necessitates heuristic sampling, which may omit optimal comparisons. Furthermore, the authors note that preference labels disproportionately improve performance on difficult samples, whereas additional task labels remain more effective for easy samples, indicating that preference learning is a complementary rather than universally superior annotation strategy.
