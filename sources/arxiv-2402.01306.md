---
id: arxiv:2402.01306
type: paper
title: Kahneman-Tversky Optimization (KTO)
url: https://arxiv.org/abs/2402.01306
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-variants
---

# Kahneman-Tversky Optimization (KTO)

Kahneman-Tversky Optimization (KTO) is a model alignment method that frames the alignment of Large Language Models (LLMs) through the lens of **prospect theory**. The authors propose that the success of existing alignment methods, such as Direct Preference Optimization (DPO), is due to their nature as **Human-Aware Losses (HALOs)**—loss functions that implicitly incorporate human cognitive biases, such as loss aversion and diminishing sensitivity.

### Core Problem
Traditional alignment methods (RLHF, DPO) rely on preference data (pairs of preferred $y_w$ and dispreferred $y_l$ outputs), which is expensive and scarce to collect. While binary signals (whether a single output $y$ is desirable or undesirable) are more abundant, they are typically viewed as weaker signals. KTO aims to leverage these binary signals to match or exceed the performance of preference-based methods by directly maximizing human utility rather than the log-likelihood of preferences.

### Method and Recipe
KTO derives its objective from the Kahneman-Tversky model of human value. To ensure numerical stability during optimization, KTO replaces the original power-law value function with a logistic function.

**Step-by-Step Implementation:**
1. **Define Implied Reward:** Calculate the reward as the log-ratio of the current policy $\pi_\theta$ to the reference model $\pi_{\text{ref}}$.
2. **Estimate Reference Point ($z_0$):** Since sampling from $\pi_\theta$ is slow, KTO uses a biased estimate $\hat{z}_0$ by shifting outputs within a microbatch of size $m$ to create mismatched pairs $\{(x_i, y_j)\}$.
3. **Apply Value Function:** Map the difference between the implied reward and the reference point to a subjective value using a logistic function, applying different weights ($\lambda_D, \lambda_U$) for desirable and undesirable outputs.
4. **Optimize Loss:** Minimize the expected difference between the target weight and the perceived value.

### Key Formulas
The implied reward is defined as:

$$
r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

The reference point $\hat{z}_0$ is estimated as:

$$
\hat{z}_0 = \max \left(0, \frac{1}{m} \sum_{1 \le i < m} \log \frac{\pi_\theta(y_j|x_i)}{\pi_{\text{ref}}(y_j|x_i)}\right)
$$

where $j = (i + 1) \mod m$.

The value function $v(x, y)$ is:

$$
v(x, y) = \begin{cases} \lambda_D \sigma(\beta(r_\theta(x, y) - z_0)) & \text{if } y \sim y_{\text{desirable}} | x \\ \lambda_U \sigma(\beta(z_0 - r_\theta(x, y))) & \text{if } y \sim y_{\text{undesirable}} | x \end{cases}
$$

The final KTO loss is:

$$
L_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x, y \sim D} [\lambda_y - v(x, y)]
$$

where $\beta$ controls risk aversion and $\lambda_y$ controls loss aversion.

### Key Quantitative Results
*   **Performance vs. DPO:** KTO matches or exceeds DPO performance across model scales from 1B to 30B parameters. In human evaluations on the OpenAssistant test set, KTO achieved a winrate of $72.9\% \pm 5.3$ compared to DPO's $62.1\% \pm 5.7$.
*   **Task-Specific Gains:** On the GSM8K mathematical reasoning benchmark, swapping DPO for KTO when aligning Zephyr-$\beta$-SFT improved performance by 13.5 points (53.5 vs 40.0).
*   **Data Efficiency:** KTO can match DPO performance while using up to 90% fewer desirable examples. In a "one-y-per-x" setup (reducing data by 72%), KTO still outperformed DPO and the official Mistral-7B-Instruct.
*   **SFT Independence:** For sufficiently large models (Llama-13B, 30B), KTO can be applied directly to the pretrained model without Supervised Finetuning (SFT) without losing quality, whereas DPO without SFT leads to rambling and hallucinations.

### Stated Limitations
*   **Underfitting Risk:** KTO may underfit if the preference data is exceptionally clean (low noise and low intransitivity), as its gradient tends to zero when rewards become extreme.
*   **Hyperparameter Sensitivity:** KTO is highly sensitive to the learning rate, typically requiring a rate 2x to 10x higher than that used for DPO.
*   **Theoretical Gap:** The value function is based on monetary gambles, which may not perfectly represent how humans perceive the quality of generated text.
*   **Reference Model Dependency:** While a reference-free variant exists, it is less performant than the standard KTO.
