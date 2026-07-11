---
id: arxiv:2402.01306
type: paper
title: 'KTO: Model Alignment as Prospect Theoretic Optimization'
url: https://arxiv.org/abs/2402.01306
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-variants
---

# KTO: Model Alignment as Prospect Theoretic Optimization

### Core Problem
Aligning Large Language Models (LLMs) typically relies on preference data (pairs of preferred $y_w$ and dispreferred $y_l$ outputs), which is expensive and scarce to collect. While methods like Direct Preference Optimization (DPO) and RLHF are effective, the authors seek to understand the inductive biases that drive their success and determine if alignment can be achieved using simpler, more abundant binary signals (whether an output is desirable or undesirable) without sacrificing performance.

### Method: Kahneman-Tversky Optimization (KTO)
The authors frame alignment through **Prospect Theory**, proposing a class of **Human-Aware Losses (HALOs)**. KTO is a HALO that directly maximizes the perceived human utility of generations rather than the log-likelihood of preferences.

#### Step-by-Step Recipe:
1.  **Define Implied Reward**: Calculate the reward as the log-ratio of the current policy $\pi_\theta$ to the reference model $\pi_{\text{ref}}$:

$$
r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

2.  **Estimate Reference Point**: To avoid slow sampling from $\pi_\theta$, KTO uses a biased estimate $\hat{z}_0$ of the KL divergence by shifting outputs within a microbatch of size $m$:

$$
\hat{z}_0 = \max \left(0, \frac{1}{m} \sum_{1 \leq i < m} \log \frac{\pi_\theta(y_j|x_i)}{\pi_{\text{ref}}(y_j|x_i)}\right)
$$

    where $j = (i + 1) \mod m$.
3.  **Apply Value Function**: Use a logistic function $\sigma$ (replacing the original power-law form for numerical stability) to determine utility $v(x, y)$ based on whether the output $y$ is desirable or undesirable:

$$
v(x, y) = \begin{cases} \lambda_D \sigma(\beta(r_\theta(x, y) - z_0)) & \text{if } y \sim y_{\text{desirable}} \\ \lambda_U \sigma(\beta(z_0 - r_\theta(x, y))) & \text{if } y \sim y_{\text{undesirable}} \end{cases}
$$

    Here, $\beta$ controls risk aversion, and $\lambda_D, \lambda_U$ control loss aversion/weighting.
4.  **Minimize Loss**: The KTO loss is defined as:

$$
L_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x, y \sim D} [\lambda_y - v(x, y)]
$$

    where $\lambda_y$ is $\lambda_D$ or $\lambda_U$ depending on the label.

### Key Quantitative Results
*   **Performance vs. DPO**: KTO matches or exceeds DPO performance across model scales from 1B to 30B parameters. On the GSM8K benchmark, replacing DPO with KTO when aligning Zephyr-$\beta$-SFT improved performance by **13.5 points**.
*   **Data Efficiency**: KTO is highly robust to data imbalance. It can match DPO performance while using up to **90% fewer desirable examples** (e.g., a 1:10 ratio of desirable to undesirable outputs).
*   **SFT Independence**: For sufficiently large models (Llama-13B and 30B), KTO can skip the Supervised Finetuning (SFT) stage without loss in generation quality. In contrast, DPO without SFT leads to significant rambling and hallucinations.
*   **Binary Signal Success**: When aligning Mistral-7B on OpenAssistant, a "one-y-per-x" KTO setup (reducing training data by 72%) achieved a winrate of **$0.631 \pm 0.036$** against SFT targets, outperforming DPO (**$0.600 \pm 0.037$**).

### Stated Limitations
*   **Underfitting Risk**: KTO may underfit if the preference data is exceptionally clean (low noise and low intransitivity), as the gradient tends to zero when rewards become extreme.
*   **Theoretical Proxy**: The value function is derived from monetary gamble experiments; the authors acknowledge it may not perfectly capture how humans perceive the relative quality of text.
*   **Hyperparameter Sensitivity**: KTO requires a significantly more aggressive learning rate than DPO (typically **2x to 10x higher**) to compensate for smaller reference-adjusted rewards.
*   **Reference Model Dependency**: While a reference-free variant exists, it is less performant than standard KTO.
