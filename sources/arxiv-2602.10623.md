---
id: arxiv:2602.10623
type: paper
title: Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling
url: https://arxiv.org/abs/2602.10623
retrieved: '2026-07-12'
maturity: comprehensive
topic: reward-modeling
---

# Summary: Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling

### Core Problem
Reinforcement Learning from Human Feedback (RLHF) relies on reward models (RMs) to align Large Language Models (LLMs). However, these RMs are susceptible to **reward hacking** (or over-optimization), where the policy exploits spurious correlations—such as response length, phrasing patterns, or stylistic artifacts—rather than genuine human intent. This is primarily caused by reward misgeneralization, stemming from noisy, subjective human annotations and the tendency of deep neural networks to learn "shortcut" features.

### Method
The authors propose the **Bayesian Non-negative Reward Model (BNRM)**, which replaces the standard deterministic reward head with a probabilistic generative process based on Non-negative Factor Analysis (NFA).

**Step-by-Step Recipe:**
1.  **Latent Factorization:** The model represents rewards using two sets of non-negative latent variables:
    *   **Local Latent Variables ($\theta$):** Instance-specific factors that induce disentangled reward representations, suppressing idiosyncratic features.
    *   **Global Reward Dictionary ($\Phi$):** A shared basis across all data points that acts as a population-level regularizer to suppress systematic spurious correlations (debiasing).
2.  **Probabilistic Priors:** Gamma priors are placed on both $\theta$ and $\Phi$ to jointly enforce non-negativity and sparsity.
3.  **Reward Generation:** The scalar reward for a response $(x, y)$ is computed as the dot product of the local factors and the global dictionary.
4.  **Preference Modeling:** The resulting rewards are fed into a Bradley-Terry (BT) likelihood function to predict human preferences.
5.  **Amortized Variational Inference:** To scale to LLMs, the authors use the LLM backbone as an inference network (encoder). This network predicts the shape ($k$) and scale ($\lambda$) parameters of a **reparameterizable Weibull distribution** to approximate the posterior distributions $q(\theta|x, y)$ and $q(\Phi)$.
6.  **Training:** The model is trained by maximizing the Evidence Lower Bound (ELBO), balancing the reconstruction likelihood of preferences against the KL divergence from the Gamma priors.

### Key Formulas
The reward $r$ for a response $(x, y)$ is defined as:

$$
r(x, y) = \theta^\top \Phi
$$

The probability that response $y_1$ is preferred over $y_2$ follows the Bradley-Terry model:

$$
p(y_1 \succ y_2 | r_1, r_2) = \sigma(r_1 - r_2)
$$

The training objective maximizes the ELBO $\mathcal{L}(\mathcal{D})$:

$$
\max \mathcal{L}(\mathcal{D}) = \max \left[ \mathbb{E}_{q(\theta, \Phi)}[\log p(\mathcal{D}|\theta, \Phi)] - \eta KL(q(\theta)||p(\theta)) - \eta KL(q(\Phi)||p(\Phi)) \right]
$$

where $\eta$ is the KL divergence coefficient.

### Key Quantitative Results
*   **Generalization:** On the Unified-Feedback (UF) dataset (40K samples), BT-BNRM achieved accuracies of **74.2% (ID)**, **83.6% (HHH Alignment)**, and **75.2% (MT-Bench)**, improving over the BT baseline by 5.4, 13.3, and 6.1 percentage points, respectively.
*   **RewardBench:** The BNBT-Reward-Llama-3.1-8B model achieved an overall score of **93.6**, outperforming the Skywork-Reward-Llama-3.1-8B baseline (93.1).
*   **Debiasing:** BNRM significantly reduced length bias; the Pearson correlation between response length and reward score dropped from **0.488 (BT)** to **0.123 (BNRM)** on the RM-Bench Hard subset.
*   **Data Efficiency & Noise Tolerance:** 
    *   BNRM trained on only **1K examples** matched the performance of a BT model trained on **20K examples** on RewardBench.
    *   Under a **40% label noise rate**, BNRM improved BT performance by up to **16.7%**.
*   **RLHF Performance:** PPO-tuned policies using BNRM reached accuracies of **74.98%** (Llama-3.1-8B-Instruct) and **62.25%** (OpenRLHF-Llama3-8B-SFT).

### Stated Limitations
1.  **Hyperparameter Sensitivity:** The model introduces a method-specific hyperparameter $\eta$ (KL coefficient). While robust, achieving peak performance requires tuning (the authors found $\eta = 10^{-5}$ to be optimal).
2.  **Scope of Evaluation:** While effective in current benchmarks, the authors note that open-ended RLHF settings may expose more diverse and adaptive forms of reward hacking, and the model has not yet been tested in multi-turn, tool-use, or long-horizon alignment scenarios.
