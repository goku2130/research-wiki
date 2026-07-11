---
id: arxiv:2309.06256
type: paper
title: Mitigating the Alignment Tax of RLHF
url: https://arxiv.org/abs/2309.06256
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

The paper addresses the "alignment tax" in Reinforcement Learning with Human Feedback (RLHF), a phenomenon where aligning Large Language Models (LLMs) with human preferences causes catastrophic forgetting of pre-trained natural language processing (NLP) capabilities. To mitigate this degradation, the authors systematically benchmark regularization, LoRA, knowledge distillation, reward penalties, and model averaging (MA). They demonstrate that MA, which simply interpolates between pre-RLHF weights $\theta_0$ and post-RLHF weights $\theta$, achieves the most efficient alignment-forgetting Pareto front. Theoretical analysis reveals that MA enhances performance by increasing feature diversity in transformer layers where NLP and alignment tasks share overlapping feature spaces, particularly in low-level layers that capture foundational representations like word embeddings.

Building on these insights, the authors propose Heterogeneous Model Averaging (HMA). The method follows a precise recipe: (1) Perform vanilla RLHF to obtain the aligned model $\theta$ and policy $\pi_\theta$. (2) Distill a high-reward dataset $\mathcal{D}_\theta$ from $\pi_\theta$ using an iterative best-of-$n$ sampling strategy. (3) Partition the transformer into $K$ contiguous parts (default $K=3$: input layers 1–8, middle 9–17, output 18–26). (4) Assign a unique averaging ratio $\alpha_i \in [0,1]$ to each part $i$. (5) Optimize the ratio vector $(\alpha_1, \dots, \alpha_K)$ to maximize alignment reward while constraining the mean ratio to a fixed $\alpha$. (6) Return the averaged model $\theta^{(K)}$.

Key mathematical formulations define the approach. The base RLHF objective maximizes expected reward: $\max_{\theta}\mathbb{E}_{x}\mathbb{E}_{a\sim\pi_{\theta}(\cdot|x)}[r^{*}(x,a)]$. Standard MA interpolates weights as $\pi_{(1-\alpha)\theta_{0}+\alpha\theta}$. HMA applies heterogeneous ratios per layer block: $\theta^{[k]}(K):=\alpha_{k}\theta^{[k]}+(1-\alpha_{k})\theta_{0}^{[k]}$. The optimization problem solves for the best ratios under the constraint $\frac{1}{K}\sum_{k}\alpha_{k}=\alpha$:

$$
\max_{(\alpha_{1},\ldots,\alpha_{K})\in\Omega}\mathbb{E}_{x}\mathbb{E}_{a\sim\pi_{\theta(K)}(\cdot|x)}\left[r^{*}(x,a)\right], \quad \text{where } \Omega = \left\{(\alpha_1,\dots,\alpha_K) \mid \frac{1}{K}\sum_{k}\alpha_{k}=\alpha, \alpha_{k}\in[0,1]\right\}.
$$

Theoretically, the authors prove that averaging improves robustness when tasks share feature spaces, showing $\xi^{(1)}-\xi^{(2)}=F_{p}\left(\frac{\sqrt{2}(1-p)n}{\sqrt{n+n_{o}}}\right)-F_{p}\left((1-p)\sqrt{n}\right)\geq0$, where $\xi$ denotes effective averaging robustness and $F_p$ is a cumulative density function.

Quantitatively, experiments span OpenLLaMA-3B and Mistral-7B variants (Zephyr-7B-β, Zephyr-7B-Gemma) aligned via Rejection Sampling Fine-tuning (RSF), Direct Preference Optimization (DPO), and PPO. Alignment tax is measured across common sense QA (ARC, Race, PIQA), reading comprehension (SQuAD, DROP), and translation (WMT 2014 Fr-En). The authors identify $\alpha=0.2$ as a robust default that consistently mitigates forgetting without degrading alignment. HMA with $K=3$ consistently outperforms vanilla MA across all benchmarks. Ablation studies show that increasing $K$ to 6 or 9 yields marginal gains but risks overfitting, slightly reducing reading comprehension performance. Evaluations using the PairRM preference model and GPT4 on AlpacaEval 2.0 confirm that HMA consistently surpasses baseline Zephyr-7B-β across all metrics.

The authors acknowledge that HMA significantly alleviates but does not fully eliminate the alignment tax. They note that experience replay methods are computationally prohibitive and less effective due to the unavailability and vast scale of pre-training datasets. Future work must determine the theoretical lower bound of alignment tax and identify methods achieving the optimal trade-off frontier.
