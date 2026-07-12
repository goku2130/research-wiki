---
id: arxiv:2502.13177
type: paper
title: KL Penalty Control via Perturbation for Direct Preference Optimization
url: https://arxiv.org/abs/2502.13177
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

# Summary: $\epsilon$-Direct Preference Optimization ($\epsilon$-DPO)

### Core Problem
Direct Preference Optimization (DPO) utilizes a static KL penalty coefficient $\beta$ to prevent the policy from deviating excessively from the reference model. However, applying a uniform $\beta$ across all preference pairs is suboptimal, as different instances may require different levels of regularization. Existing attempts to dynamicize $\beta$ (e.g., $\beta$-DPO) operate at the batch level rather than the instance level, and other methods (e.g., TR-DPO) update the reference policy periodically without adaptive, instance-specific control.

### Method
$\epsilon$-DPO introduces an instance-level adaptive KL penalty control mechanism based on the monotonicity of logits under the perturbation of $\beta$. The process is as follows:

1.  **Perturbation Definition**: For a given $\beta$ and a small positive constant $\epsilon$, two perturbed values are defined:

$$
\beta_{\varepsilon}^{-} := \frac{\beta}{1+\varepsilon}, \quad \beta_{\varepsilon}^{+} := \frac{\beta}{1-\varepsilon}
$$

2.  **Policy Estimation**: To avoid training multiple models, $\epsilon$-DPO approximates the policies under perturbed $\beta$ by reusing the logits of the current policy $f_\theta$ and the reference policy $f_{\text{ref}}$. Following the approximation that the optimal policy under $\beta/\lambda$ is the arithmetic mean of logits:

$$
\pi_{\theta(\beta_{\varepsilon}^{-})} \approx \prod_{i=1}^{n} \text{Softmax}\big((1+\varepsilon)f_{\theta}(x,y_{1:i-1}) - \varepsilon f_{\text{ref}}(x,y_{1:i-1})\big)_{y_{i}}
$$

3.  **Monotonicity Criterion**: The log-likelihood ratio (logit) for a preference triplet $(x, y^w, y^l)$ is calculated as $z_{\theta}(x, y^w, y^l) = \log \frac{\pi_\theta(y^w|x)}{\pi_\theta(y^l|x)}$. The instance-level coefficient $\tilde{\beta}$ is determined by:
    *   $\tilde{\beta} = \beta_{\varepsilon}^{-}$ if $z_{\theta(\beta_{\varepsilon}^{-})} > z_{\theta(\beta)} > z_{\theta(\beta_{\varepsilon}^{+})}$
    *   $\tilde{\beta} = \beta_{\varepsilon}^{+}$ if $z_{\theta(\beta_{\varepsilon}^{-})} < z_{\theta(\beta)} < z_{\theta(\beta_{\varepsilon}^{+})}$
    *   $\tilde{\beta} = \beta$ otherwise.
4.  **Optimization and Update**: The model is updated using the DPO loss $\mathcal{L}_{\text{DPO}}$ with the instance-specific $\tilde{\beta}$. After the update, the baseline $\beta$ for the next step is updated as the mean of the determined $\tilde{\beta}$ across the batch: $\beta \leftarrow \mathbb{E}_{x,y^w,y^l}[\tilde{\beta}]$.

### Key Formulas
The DPO objective used for alignment is:

$$
\mathcal{L}_{\text{DPO}}(x,y^{w},y^{l};\theta,\beta) := -\log \sigma \left( \beta \log \frac{\pi_{\theta}(y^{w}|x)}{\pi_{\text{ref}}(y^{w}|x)} - \beta \log \frac{\pi_{\theta}(y^{l}|x)}{\pi_{\text{ref}}(y^{l}|x)} \right)
$$

### Key Quantitative Results
$\epsilon$-DPO was evaluated on Mistral-7B-Instruct and Llama-3-8B-Instruct using UltraFeedback.

*   **General Performance (Llama-3-Instruct)**: $\epsilon$-DPO outperformed DPO and several direct alignment algorithms (e.g., SimPO, IPO, CPO) across multiple benchmarks:
    *   **AlpacaEval 2 (Length-Controlled Win Rate)**: 46.4% (compared to DPO's 40.3%).
    *   **Arena-Hard (Win Rate)**: 36.7% (compared to DPO's 32.6%).
    *   **MT-Bench (Score)**: 8.0.
*   **Computational Overhead**: The additional computation is negligible. The wall-time increment $\Delta t$ per training step was $0.0008\text{s}$ for Mistral-Instruct and $0.0006\text{s}$ for Llama-3-Instruct.
*   **KL Trade-off**: On the Anthropic-HH dataset, $\epsilon$-DPO achieved a superior Pareto frontier between forward KL divergence and win rate compared to TR-DPO, indicating more efficient KL trade-off.

### Stated Limitations
*   **Sensitivity to $\epsilon$**: The approximation of the optimal policy can be unstable at high $\epsilon$ levels, particularly in the early stages of training.
*   **Downstream Task Degradation**: Consistent with other direct alignment algorithms, $\epsilon$-DPO exhibits a performance drop in specific downstream tasks, specifically math skills (GSM8K), as seen in the Open LLM Leaderboard results.
