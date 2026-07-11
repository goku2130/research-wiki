---
id: arxiv:2602.10623
type: paper
title: Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling
url: https://arxiv.org/abs/2602.10623
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-hacking
---

# Bayesian Non-negative Reward Modeling (BNRM)

### Core Problem
In Reinforcement Learning from Human Feedback (RLHF), reward models (RMs) are prone to **reward hacking** (or reward over-optimization). This occurs when a policy exploits spurious correlations—such as response length, stylistic artifacts, or phrasing patterns—encoded in the proxy RM rather than the true human objective. This failure is driven by noisy, subjective human annotations and the tendency of deep neural networks to rely on "shortcut" features, leading to poor generalization under distribution shifts.

### Method
The Bayesian Non-negative Reward Model (BNRM) mitigates reward hacking by replacing deterministic reward functions with a structured, sparsity-aware Bayesian generative process based on Non-negative Factor Analysis (NFA).

**Step-by-Step Recipe:**
1.  **Latent Factor Representation:** Instead of a scalar output, the reward is decomposed into two non-negative latent variables:
    *   **Local Latent Variables ($\theta$):** Instance-specific factors that induce disentangled representations of semantic preferences.
    *   **Global Reward Dictionary ($\Phi$):** A shared basis across all data points. Sparsity in $\Phi$ acts as a population-level regularizer to suppress non-causal biases.
2.  **Prior Specification:** Gamma priors are placed on both $\theta$ and $\Phi$ to jointly enforce non-negativity and sparsity.
3.  **Reward Generation:** The scalar reward for a prompt-response pair $(x, y)$ is computed as the dot product of the local and global factors.
4.  **Preference Modeling:** The observed human preference between two responses ($y_1, y_2$) is modeled using the Bradley-Terry (BT) likelihood.
5.  **Amortized Variational Inference:** To scale to LLMs, a deep backbone (e.g., Gemma or Llama) acts as an inference network (encoder). The variational posteriors $q(\theta|x,y)$ and $q(\Phi)$ are parameterized using a reparameterizable **Weibull distribution**, enabling end-to-end training via backpropagation.
6.  **Optimization:** The model is trained by maximizing the Evidence Lower Bound (ELBO), which balances the reconstruction likelihood of preferences against the KL divergence from the Gamma priors.

### Key Formulas
The scalar reward is defined as:

$$
r(x,y) = \theta^\top \Phi
$$

The probability of preference is given by the Bradley-Terry model:

$$
p(y_1 \succ y_2 \mid r_1, r_2) = \sigma(r_1 - r_2)
$$

The training objective (ELBO) is:

$$
\max \mathcal{L}(\mathcal{D}) = \max \left[ \mathbb{E}_{q(\theta|x,y)q(\Phi)}[\log p(\mathcal{D} \mid \theta, \Phi)] - \eta KL(q(\theta|x,y) \parallel p(\theta)) - \eta KL(q(\Phi) \parallel p(\Phi)) \right]
$$

where $\eta$ is a trade-off coefficient balancing likelihood and regularization.

### Quantitative Results
*   **Robustness and Generalization:** On the Unified-Feedback (UF) 40K dataset using a Gemma-2B-it backbone, BT-BNRM improved accuracy over the BT baseline by **5.4% on UF**, **13.3% on HHH Alignment**, **6.1% on MT-Bench**, and **8.0% on RewardBench**.
*   **High-Capacity Performance:** When refining Skywork-Reward-Llama-3.1-8B, BNRM achieved an overall RewardBench score of **93.6**, outperforming the original model (93.1).
*   **RLHF Alignment:** PPO-tuned policies using BNRM achieved accuracies of **74.98%** (Llama-3.1-8B-Instruct) and **62.25%** (OpenRLHF-Llama3-8B-SFT). On Arena-Hard-v0.1, the BNRM-aligned model achieved a **50% win rate** and **28% tie rate** against the base Llama-3.1-8B-Instruct.
*   **Debiasing:** On the RM-Bench Hard subset (designed to test length bias), BNRM reduced the Pearson correlation between response length and reward from **0.488 (BT)** to **0.123**.
*   **Data Efficiency and Noise Tolerance:** BNRM trained on only **1K examples** matched the RewardBench performance of a BT model trained on **20K**. Under 40% label noise, BNRM improved BT performance by up to **16.7%**.

### Limitations
*   **Hyperparameter Sensitivity:** The model introduces the KL coefficient $\eta$, which requires tuning to balance prior regularization and posterior adaptation (optimal $\eta = 10^{-5}$ in experiments).
*   **Evaluation Scope:** While effective on proxy scores and specific benchmarks, the authors note that open-ended RLHF may expose more adaptive forms of reward hacking.
*   **Scenario Constraints:** The framework has not yet been extensively evaluated in multi-turn, tool-use, or long-horizon alignment scenarios.
