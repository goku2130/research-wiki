---
id: arxiv:2210.10760
type: paper
title: Scaling Laws for Reward Model Overoptimization
url: https://arxiv.org/abs/2210.10760
retrieved: '2026-07-12'
maturity: comprehensive
topic: reward-model-overoptimization
---

# Scaling Laws for Reward Model Overoptimization

### Core Problem
In Reinforcement Learning from Human Feedback (RLHF), a proxy reward model (RM) is trained to predict human preferences. Because this proxy is an imperfect representation of the true objective, excessive optimization against it can lead to a decline in actual performance—a phenomenon known as **overoptimization** or **Goodhart’s Law**. The authors seek to empirically measure this effect and determine how it scales with reward model parameters, dataset size, policy size, and optimization methods.

### Methodology
To avoid the expense of human labeling, the authors employ a synthetic setup where a "gold-standard" RM (a 6B parameter model) acts as the ground truth. 

1.  **Data Generation**: The gold RM labels pairs of rollouts from a policy; the trajectory with the higher gold score is marked as preferred. 100,000 synthetic comparisons are generated.
2.  **Proxy RM Training**: Proxy RMs (ranging from 3M to 3B parameters) are trained on these synthetic labels.
3.  **Optimization**: The authors optimize the policy against the proxy RM using two methods:
    *   **Best-of-$n$ (BoN) Sampling**: Generating $n$ trajectories and selecting the one with the highest proxy score.
    *   **Reinforcement Learning (RL)**: Using Proximal Policy Optimization (PPO).
4.  **Measurement**: The gold RM score $R$ is measured as a function of the distance $d$, defined as the square root of the Kullback–Leibler (KL) divergence from the initial policy $\pi_{\mathrm{init}}$ to the optimized policy $\pi$:

$$
d = \sqrt{D_{\mathrm{KL}}(\pi \parallel \pi_{\mathrm{init}})}
$$

    For BoN, KL is computed analytically as $\mathrm{KL}_{\mathrm{bon}} = \log n - \frac{n-1}{n}$.

### Key Formulas
The authors find that the gold reward $R$ follows distinct functional forms based on the optimization method:

*   **Best-of-$n$ Sampling**:

$$
R_{\mathrm{bon}}(d) = d(\alpha_{\mathrm{bon}} - \beta_{\mathrm{bon}}d)
$$

*   **Reinforcement Learning**:

$$
R_{\mathrm{RL}}(d) = d(\alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}}\log d)
$$

In these equations, $\alpha$ and $\beta$ are coefficients that vary based on the proxy RM's size and training data.

### Key Quantitative Results
*   **RM Parameter Scaling**: The coefficients $\alpha$ and $\beta$ scale smoothly and approximately logarithmically with the number of proxy RM parameters. This allows for the prediction of peak gold RM scores for different RM sizes.
*   **RM Data Scaling**: A critical threshold exists for RM training data; for datasets smaller than approximately **2,000 comparisons**, there is very little improvement over near-chance loss. Beyond this, gold scores improve with more data.
*   **Policy Size**: Larger policies (e.g., 6B vs 1.2B) start with higher initial gold scores and see less relative benefit from optimization. However, they do not overoptimize faster; the gold scores peak at nearly the same KL distance, and the gap between proxy and gold scores remains similar.
*   **RL vs. BoN**: RL is significantly less "KL-efficient" than BoN, consuming far more KL distance to achieve the same level of optimization or overoptimization. However, the relationship between proxy and gold scores is similar across both methods.
*   **KL Penalty**: In RL, adding a KL penalty does not improve the gold RM score-KL frontier; it merely causes the gold reward to converge earlier, acting similarly to early stopping.

### Stated Limitations
*   **Synthetic Setup**: The use of a gold RM may not perfectly transfer to real-world settings where human preferences are the ground truth.
*   **Human Intent Gap**: The study does not capture overoptimization resulting from the mismatch between ground truth labels and actual human intent (e.g., labelers choosing options that only *appear* correct).
*   **Scope of Policy Scaling**: Policy size exploration was limited to only two different sizes.
*   **Adversarial Goodhart**: The models used were not powerful enough to implement active adversarial strategies to manipulate the proxy RM; such capabilities in future models may break the observed scaling laws.
