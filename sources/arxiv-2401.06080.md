---
id: arxiv:2401.06080
type: paper
title: 'Secrets of RLHF in Large Language Models Part II: Reward Modeling'
url: https://arxiv.org/abs/2401.06080
retrieved: '2026-07-10'
maturity: comprehensive
topic: ppo-for-llms
---

**Core Problem**
Reward models (RMs) in RLHF serve as proxies for human preferences but face two critical bottlenecks: (1) preference datasets contain incorrect and ambiguous pairs due to low annotator agreement, hindering accurate intent capture; and (2) RMs trained on specific data distributions generalize poorly to out-of-distribution (OOD) examples, destabilizing iterative RLHF optimization.

**Method/Recipe**
The authors propose a dual-pronged approach addressing data quality and algorithmic generalization. First, preference strength is quantified by training an ensemble of reward models with randomized training orders and computing the mean ($\mu$) and standard deviation ($\sigma$) of their score differences for each prompt-response pair. Data is partitioned by strength into incorrect (lowest strength), ambiguous (near-zero strength), and normal/strong (highest strength) categories. To mitigate noise, labels for the bottom-performing data are flipped, while label smoothing is applied to ambiguous and strong preference data to prevent overconfidence. An adaptive margin is added to the loss function, scaling with preference strength to enforce larger score gaps for easily distinguishable pairs. Second, to enhance generalization, unsupervised contrastive learning is integrated to sharpen discrimination of subtle response differences, and meta-learning is employed to align the preference data distribution with the model’s output distribution, enabling OOD transfer and iterative RLHF updates. The cleaned reward model subsequently drives PPO fine-tuning, where the KL penalty can be omitted without destabilizing training.

**Key Formulas**
The preference distribution follows the Bradley-Terry model:
$$P(y_w | x, y_l, y_w) = \frac{\exp(R(x, y_w))}{\exp(R(x, y_w)) + \exp(R(x, y_l))}$$
Training minimizes the negative log-likelihood:
$$\mathcal{L} = -\log P(y_w | x, y_l, y_w)$$
The RL objective maximizes reward while penalizing divergence from the reference policy:
$$\max_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta} [R(x, y)] - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]$$
Preference strength is computed across $N$ ensemble models as:
$$\mu = \frac{1}{N}\sum_{i=1}^{N} R_i, \quad \sigma = \text{std}(R_i)$$
Label smoothing modifies the target distribution to $q = (1-\epsilon)y + \epsilon/2$, minimizing cross-entropy between $q$ and model outputs $p$. Finally, the adaptive margin loss incorporates a continuous strength measure:
$$\mathcal{L}_{total} = \mathcal{L}_{NLL} + \lambda \cdot \text{margin}(\mu)$$

**Key Quantitative Results**
Using an ensemble of 10 reward models, the method partitions training data by preference strength. Experiments demonstrate that training exclusively on the bottom 10% of data negatively impacts validation performance, while the middle 40% yields approximately 80% prediction accuracy. The top 10% of strong-preference data significantly improves performance but exhibits rapid training loss decay, indicating overfitting risk. Denoising strategies (label flipping, smoothing, and adaptive margins) stabilize PPO optimization, producing linear KL divergence growth compared to the baseline’s rapid increase and fluctuations. GPT-4 evaluations confirm substantial improvements in harmlessness, though gains for helpfulness are more modest.

**Stated Limitations**
The approach acknowledges that original validation sets inherently contain noise, which can mislead RM training if unaddressed. The reliance on GPT-4 for supplementary validation annotations is noted as imperfect. Furthermore, while denoising effectively mitigates harmful prompt vulnerabilities, the model exhibits less pronounced improvement on helpful prompts, indicating unresolved conflicts between optimizing for harmlessness and helpfulness. The contrastive learning and meta-learning components are introduced theoretically to bridge distribution gaps, but their full empirical integration and iterative RLHF scaling are noted as ongoing extensions.
