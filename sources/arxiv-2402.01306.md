---
id: arxiv:2402.01306
type: paper
title: 'Aligning AI with Shared Human Preferences: Interactions via Constraint Optimization
  and Collaborative Inference'
url: https://arxiv.org/abs/2402.01306
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-variants
---

**Core Problem**
Aligning large language models (LLMs) with human feedback has historically depended on preference data (paired preferred and dispreferred outputs), which is resource-intensive and scarce. The authors investigate whether preference pairs are strictly necessary by framing alignment through prospect theory, demonstrating that successful objectives like Direct Preference Optimization (DPO) implicitly model human cognitive biases such as loss aversion. They formalize this observation as human-aware losses (HALOs) and propose Kahneman-Tversky Optimization (KTO), a method that aligns models using only binary desirable/undesirable signals while directly maximizing human utility rather than maximizing the log-likelihood of preferences.

**Method and Recipe**
KTO derives a HALO from the Kahneman-Tversky value function, replacing the original power-law exponent with a logistic function for numerical stability. The training recipe proceeds as follows: (1) Compute the implied reward $r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$, where $\pi_{\text{ref}}$ is a frozen reference model. (2) Define the reference point $z_0$ as the KL divergence $\text{KL}(\pi_\theta(y'|x) \| \pi_{\text{ref}}(y'|x))$. In practice, $z_0$ is estimated stably within a microbatch of size $m$ using mismatched outputs to avoid backpropagation through the KL term: $\hat{z}_0 = \max\left(0, \frac{1}{m} \sum_{1 \leq i < m} \log \frac{\pi_\theta(y_j|x_i)}{\pi_{\text{ref}}(y_j|x_i)}\right)$, where $j = (i+1) \mod m$. (3) Apply hyperparameters $\beta$ (risk aversion) and $\lambda_D, \lambda_U$ (loss aversion for desirable/undesirable outputs) to shape the value function $v(x,y)$. (4) Optimize the KTO loss over the dataset $\mathcal{D}$. (5) Decompose preference pairs $(y_w \succ y_l)$ into binary signals, assuming $y_w$ is desirable and $y_l$ is undesirable. (6) Mitigate class imbalance by scaling $\lambda_D$ and $\lambda_U$ such that $\frac{\lambda_D n_D}{\lambda_U n_U} \in [1, 1.5]$.

**Key Formulas**
The canonical Kahneman-Tversky value function is adapted as:
$$v(z; \lambda, \alpha, z_0) = \begin{cases} (z - z_0)^\alpha & \text{if } z \ge z_0 \\ -\lambda(z_0 - z)^\alpha & \text{if } z < z_0 \end{cases}$$
The KTO loss function is defined as:
$$L_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x,y \sim \mathcal{D}} [\lambda_y - v(x, y)]$$
where the value function $v(x,y)$ uses a logistic approximation:
$$v(x, y) = \begin{cases} \lambda_D \sigma(\beta(r_\theta(x, y) - z_0)) & \text{if } y \sim y_{\text{desirable}}|x \\ \lambda_U \sigma(\beta(z_0 - r_\theta(x, y))) & \text{if } y \sim y_{\text{undesirable}}|x \end{cases}$$
HALOs are formally characterized by a value function $v$ that is non-decreasing and concave in gains, with the general loss structure:
$$f(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x,y \sim \mathcal{D}}[a_{x,y} v(r_\theta(x, y) - \mathbb{E}_Q[r_\theta(x, y')])] + C_{\mathcal{D}}$$

**Quantitative Results**
KTO matches or exceeds DPO performance across model scales from 1B to 30B parameters. It demonstrates robustness to extreme data imbalance, matching DPO while utilizing up to 90% fewer desirable examples. For sufficiently large models (13B and 30B), KTO can be applied directly without supervised finetuning (SFT) without sacrificing generation quality, a capability DPO lacks. On the UltraFeedback dataset, replacing DPO with KTO on Zephyr-$\beta$-SFT improved GSM8K scores by 13.5 points. Even when trained on a single output per input (reducing data volume by 72%), KTO-aligned Mistral-7B models outperformed both DPO and the official instruction-tuned baseline. Recommended hyperparameters include a learning rate of $5 \times 10^{-6}$, $\beta \in [0.01, 0.10]$ for SFT-aligned models, and $\beta \in [0.10, 1.00]$ for direct KTO training.

**Stated Limitations**
The authors emphasize that no single HALO is universally optimal; performance depends on matching the loss's inductive biases to the specific alignment setting. KTO may underfit complex distributions because its gradient vanishes as implied rewards approach $\pm\infty$, causing it to ignore noisy or hard-to-learn data. The value function is adapted from monetary gamble experiments, which may not accurately model human utility for text generation. Additionally, KTO resolves contradictory feedback by defaulting to the majority preference when $\lambda_D = \lambda_U$, which may conflict with fairness principles. Finally, the method requires careful hyperparameter tuning, particularly for learning rate and risk aversion, and currently assumes a naive decomposition of preference pairs into binary signals.
