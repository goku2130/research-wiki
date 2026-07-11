---
id: arxiv:2309.06256
type: paper
title: Mitigating the Alignment Tax of RLHF
url: https://arxiv.org/abs/2309.06256
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

# Mitigating the Alignment Tax of RLHF

### Core Problem
The "alignment tax" refers to the phenomenon where Reinforcement Learning from Human Feedback (RLHF) causes Large Language Models (LLMs) to forget or regress in diverse abilities acquired during pre-training—such as reasoning, translation, and common sense question-answering—while improving alignment with human preferences (helpfulness, honesty, and harmlessness). This creates a fundamental alignment-forgetting trade-off: as the model achieves higher alignment rewards, its performance on general NLP benchmarks typically declines.

### Method and Recipe
The authors first compared several techniques to mitigate this tax, including early stopping, weight-space regularization ($L_1, L_2$), Low-Rank Adaptation (LoRA), knowledge distillation, and experience replay. They found that **Model Averaging (MA)**—the simple interpolation of weights between the model before RLHF ($\theta_0$) and after RLHF ($\theta$)—achieved the most efficient Pareto front.

Building on the observation that different transformer layers exhibit distinct alignment-forgetting trade-offs, the authors proposed **Heterogeneous Model Averaging (HMA)**. The recipe is as follows:

1.  **Partitioning:** Divide the transformer into $K$ parts (default $K=3$).
2.  **Weight Interpolation:** Assign a unique averaging ratio $\alpha_i \in [0, 1]$ to each part $i$. The weights for the $k$-th part of the merged model $\theta^{(K)}$ are calculated as:

$$
\theta^{[k](K)} := \alpha_k \theta^{[k]} + (1 - \alpha_k) \theta_0^{[k]}
$$

3.  **Optimization:** To maximize alignment reward while maintaining a forgetting level comparable to vanilla MA, the ratios $(\alpha_1, \dots, \alpha_K)$ are optimized under the constraint that their mean equals a target ratio $\alpha$:

$$
\max_{(\alpha_1, \dots, \alpha_K) \in \Omega} \mathbb{E}_x \mathbb{E}_{a \sim \pi_{\theta(K)}(\cdot|x)} [r^*(x, a)]
$$

    where $\Omega = \{ \frac{1}{K} \sum_{k=1}^K \alpha_k = \alpha, \alpha_k \in [0, 1] \}$.
4.  **Implementation:** The optimization is performed by distilling a dataset $\mathcal{D}_\theta$ from the RLHF model $\pi_\theta$ and maximizing the likelihood of the merged model $\pi_{\theta(K)}$ on that data.

### Theoretical Insights
The authors suggest that MA enhances the Pareto front by increasing feature diversity in layers where tasks share overlapped feature spaces. Specifically, low-level transformer layers capture general features (e.g., word representations) that benefit both NLP tasks and alignment. Averaging these layers reduces the probability of simultaneous feature failure, thereby improving both alignment reward and general performance.

### Key Quantitative Results
The method was validated using OpenLLaMA-3B, Mistral-7B, and Zephyr-7B-$\beta$ across various RLHF algorithms, including Rejection Sampling Finetuning (RSF) and Direct Preference Optimization (DPO).

*   **Optimal Ratio:** An averaging ratio of $\alpha = 0.2$ was found to consistently alleviate the alignment tax without significantly hurting alignment performance across various benchmarks.
*   **HMA Performance:** HMA consistently improved the alignment-forgetting Pareto front over vanilla MA.
*   **Model Scaling:** On Zephyr-7B-$\beta$, HMA improved win rates against GPT-4 and performance on NLP tasks compared to the vanilla model.
*   **Ablation on $K$:** While HMA remains superior to MA for $K \in \{3, 6, 9\}$, increasing $K$ beyond 3 led to a slight decrease in the trade-off curve (e.g., a drop in reading comprehension performance), which the authors attribute to overfitting.

### Limitations
The authors state that while HMA significantly alleviates the alignment tax, it does not fully eliminate it. They suggest that future research should explore the theoretical lower bound of the alignment tax to determine the optimal possible trade-off.
