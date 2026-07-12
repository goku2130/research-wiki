---
id: arxiv:2504.14177
type: paper
title: 'Direct Advantage Regression: Aligning LLMs with Online AI Reward'
url: https://arxiv.org/html/2504.14177v1
retrieved: '2026-07-12'
maturity: comprehensive
topic: rlaif
---

## Direct Advantage Regression (DAR): Aligning LLMs with Online AI Reward

This paper introduces Direct Advantage Regression (DAR), an alignment algorithm designed to leverage online AI reward for optimizing Large Language Models (LLMs) through weighted supervised fine-tuning. It aims to simplify the alignment process compared to traditional Reinforcement Learning from Human Feedback (RLHF) methods while maintaining theoretical consistency and improving efficiency.

### Core Problem

The core problem addressed is the inefficient and complex alignment of LLMs using online AI feedback. Existing Online AI Feedback (OAIF) methods, which replace human annotators with AI, primarily rely on binary preference signals. This approach discards more fine-grained AI supervision, such as preference margins or equivalences, leading to a loss of detailed task understanding. Furthermore, off-policy Direct Alignment from Preference (DAP) methods used in OAIF are not optimally adapted for iterative online learning. Traditional online RLHF methods like PPO, while effective, suffer from high implementation complexity due to requirements like an additional value model.

### Method/Recipe Step by Step

DAR operates within an expectation-maximization framework, iteratively improving the LLM policy by optimizing a weighted supervised fine-tuning (SFT) loss. The process involves:

1.  **Initialization**: Initialize the learning policy $\pi_\theta$ and the current sampling policy $\pi_{t=0}$ to the reference model $\pi_{\text{ref}}$.
2.  **Iterative Training Loop (for $t = 0$ to $T-1$)**:
    a.  **Prompt Sampling**: Sample a prompt $x$ from the prompt dataset $\mathcal{D}(x)$.
    b.  **Response Generation**: Sample $K$-shot responses $\{y_i\}_{i=1}^K$ from the current policy $\pi_t(\cdot|x)$.
    c.  **Advantage Calculation**: Calculate the advantage $A(x, y_i)$ for each sampled response using the reward model $r$. The advantage is defined as the response's reward minus the average reward of all $K$ sampled responses:
        $A(x, y_i) = r(x, y_i) - \frac{1}{K}\sum_{i=1}^K r(x, y_i)$.
    d.  **Advantage Normalization**: Apply batch-based normalization to the advantages: $A_{\text{norm}}(x, y_i) = [A(x, y_i) - \mu_A] / \sigma_A$.
    e.  **Weight Calculation**:
        *   **Advantage Weight**: $w_{\text{adv}}^i = \exp\left(\frac{1}{\alpha+\beta}A_{\text{norm}}(x, y_i)\right)$.
        *   **Regularization Weight**: $w_{\text{reg}}^i = \left(\frac{\pi_{\text{ref}}(y_i|x)}{\pi_t(y_i|x)}\right)^{\frac{\alpha}{\alpha+\beta}}$.
    f.  **Weight Clipping**: Apply a clip threshold $w_{\text{clip}}$ to the combined weight: $w_{\text{DAR}}^i = \min(w_{\text{reg}}^i \cdot w_{\text{adv}}^i, w_{\text{clip}})$.
    g.  **Policy Update**: Update the model parameters $\theta_t$ to $\theta_{t+1}$ by minimizing the weighted SFT loss. The gradient is given by:
        $\nabla_\theta \mathcal{L}_{\text{DAR}}(\pi_\theta) = -\frac{1}{K} \sum_{i=1}^K [w_{\text{DAR}}^i \nabla_\theta \log \pi_\theta(y_i|x)]$.
    h.  **Update Current Policy**: Set $\pi_{t+1} = \pi_{\theta_{t+1}}$.

### Key Formulas in LaTeX

1.  **Dual-constrained policy improvement objective**:

$$
\mathcal{J}_{\mathsf{D A R}}(\pi_{\theta};\pi_{\mathsf{r e f}},\pi_{t})=\text{m a x}_{\pi_{\theta}}\mathbb{E}_{x\sim d_{\pi_{t}}(x),y\sim\pi_{\theta}(y|x)}[A(x,y)] - \alpha\mathbb{D}_{\mathsf{K L}}\Big[\pi_{\theta}(y|x)\parallel\pi_{\mathsf{r e f}}(y|x)\Big]-\beta\mathbb{D}_{\mathsf{K L}}\Big[\pi_{\theta}(y|x)\parallel\pi_{t}(y|x)\Big]
$$

    where $\alpha$ and $\beta$ are regularization coefficients for the reference policy and current sampling policy, respectively.

2.  **Optimal policy $\pi^*$ (closed-form solution to the objective)**:

$$
\pi^{*}=\frac{1}{Z(x)}\pi_{\mathsf{r e f}}(y|x)^{\frac{\alpha}{\alpha+\beta}}\pi_{t}(y|x)^{\frac{\beta}{\alpha+\beta}}\text{e x p}\left(\frac{A(x,y)}{\alpha+\beta}\right)
$$

    where $Z(x) = \sum_y \pi_{\mathsf{r e f}}(y|x)^{\frac{\alpha}{\alpha+\beta}}\pi_{t}(y|x)^{\frac{\beta}{\alpha+\beta}}\text{e x p}\left(\frac{A(x,y)}{\alpha+\beta}\right)$ is the partition function.

3.  **Optimization objective for iterative policy regression**:

$$
\pi_{t+1}=\text{\text{a r g}\text{m a x}}_{\pi_{\theta}}\mathbb{E}_{(x,y)\sim\mathcal{D}_{\pi_{t}}} \left[\left(\frac{\pi_{\mathsf{r e f}}(y|x)}{\pi_{t}(y|x)}\right)^{\frac{\alpha}{\alpha+\beta}}\text{e x p}\left(\frac{A(x,y)}{\alpha+\beta}\right)\text{l o g}\pi_{\theta}(y|x)\right]
$$

    where $\mathcal{D}_{\pi_t}$ is the online dataset of prompt-response pairs collected by $\pi_t$.

### Key Quantitative Results and Numbers

*   **AI Reward vs. AI Preference Agreement**: AI reward consistently achieves higher human-AI agreement than AI preference across various LLM annotators (Qwen2, Llama-3, Mistral, Gemma-2, GPT-4) and tasks (TL;DR, Helpfulness, Harmlessness). For example, on TL;DR, Qwen2-72B-Instruct showed 74.97% agreement for AI reward vs. 71.35% for AI preference. Llama-3.1-405B showed 79.32% for AI reward vs. 72.76% for AI preference on TL;DR.
*   **Performance on Online Alignment Tasks (GPT-4-Turbo judged win rate over reference)**:
    *   **TL;DR**: DAR achieved a reference win rate of **98.27% ± 0.55%**, outperforming offline DPO (67.17% ± 1.91%), online AI preference methods (DPO: 78.47% ± 1.46%, IPO: 76.33% ± 0.21%, SLiC: 78.29% ± 0.96%), and online AI reward methods (RLOO: 80.23% ± 0.35%, PPO: 65.87% ± 5.23%, SFT+BEST-OF-N: 98.07% ± 0.51%).
    *   **Helpfulness**: DAR achieved **92.67% ± 1.05%**, outperforming offline DPO (81.34% ± 0.91%), online AI preference methods (DPO: 89.77% ± 0.58%, IPO: 79.74% ± 1.22%, SLiC: 90.86% ± 0.21%), and online AI reward methods (RLOO: 88.33% ± 0.25%, PPO: 72.86% ± 1.84%, SFT+BEST-OF-N: 88.26% ± 0.66%).
    *   **Harmlessness**: DAR achieved **85.84% ± 0.36%**, outperforming offline DPO (77.91% ± 0.87%), online AI preference methods (DPO: 83.55% ± 0.66%, IPO: 84.89% ± 0.29%, SLiC: 83.99% ± 0.85%), and online AI reward methods (RLOO: 84.59% ± 1.07%, PPO: 82.19% ± 0.50%, SFT+BEST-OF-N: 84.37% ± 0.54%).
*   **MT-Bench Score (GPT-4 judged)**: For fine-tuning on Helpsteer2, DAR achieved an MT-Bench score of **8.526 ± 0.066**, outperforming RLOO (8.502 ± 0.076) and SFT+BEST-OF-N (8.415 ± 0.019).
*   **Online Annotation Efficiency**: Online RLHF methods learning from AI reward require significantly less online annotations (3-5 times fewer) than online DAP methods learning from AI preference.
*   **Positional Bias in AI Preference**: LLMs acting as preference annotators showed a higher probability of selecting responses in the second position within a preference prompt. Human-AI agreement increased from 63.03% to 69.30% when the ground-truth chosen response was moved from the first to the second position.
*   **Ablation Studies**:
    *   DAR exhibits robust performance across various total regularization values, with 0.05 yielding the best performance on Helpfulness.
    *   Increasing the alpha ratio (ratio of $\alpha$ to $\alpha+\beta$) constrains the LLM more to the reference distribution, leading to more conservative behavior (shorter responses, lower win rates).
    *   DAR shows robust performance across varying Monte Carlo sampling sizes, even with a sampling size of 1.
    *   A weight clipping threshold of 20 yielded the best results.

### Stated Limitations

*   **PPO Hyperparameter Sensitivity**: The performance of PPO was found to be highly sensitive to hyperparameter selection, particularly the need for a No-EOS penalty to ensure valid responses. This led to suboptimal results in some experiments, and PPO was not applied to the Helpsteer2 dataset due to these challenges.
*   **LLM Positional Bias**: LLM annotators exhibit a positional bias when evaluating preferences, favoring responses placed closer to the ending prompt. This suggests potential challenges in LLMs' long-context understanding abilities.
*   **Absence of Human-Annotated Reward Labels**: For the three primary datasets (TL;DR, Helpfulness, Harmlessness), human-annotated reward labels were unavailable, necessitating the use of GPT-4, LLaMA-3.1-405B, and a pre-trained reward model as ground truth baselines for evaluating AI reward granularity.
