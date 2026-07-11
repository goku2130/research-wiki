---
id: arxiv:2604.01597
type: paper
title: 'Learning from the Right Rollouts: Data Attribution for PPO-based LLM Post-Training'
url: https://arxiv.org/abs/2604.01597
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

# Learning from the Right Rollouts: Data Attribution for PPO-based LLM Post-Training

### Core Problem
Traditional Proximal Policy Optimization (PPO) for Large Language Model (LLM) post-training operates on the assumption that all episodes in a generated rollout buffer provide beneficial optimization signals. However, these buffers often contain "noisy" or "unfaithful" reasoning—where a model arrives at a correct answer through flawed logic—and redundant episodes that the model has already mastered. These samples introduce noise, degrade policy updates, and cause computational inefficiency.

### Method: Influence-Guided PPO (I-PPO)
I-PPO integrates local data attribution into the RL loop to identify and eliminate episodes that are anti-aligned with a high-quality validation set. The process follows these steps:

1.  **Rollout Generation**: The actor model $\pi_\theta$ samples $n$ independent Chain-of-Thought (CoT) responses for a batch of prompts. Each episode $z_i$ is constructed as a tuple containing the prompt, response, total reward, value estimates, and Generalized Advantage Estimation (GAE) advantages.
2.  **Validation Gradient Calculation**: The framework computes an average validation gradient $\bar{g}_{val}$ based on the Supervised Fine-Tuning (SFT) loss over a validation set $\mathcal{D}_{val}$:

$$
\bar{g}_{val} = \nabla_\theta \mathcal{L}_{SFT}(\mathcal{D}_{val}, \theta_\tau)
$$

3.  **Episode Gradient Calculation**: For each episode $z_i$ in the rollout buffer, the gradient of the PPO loss is computed:

$$
g_{train} = \nabla_\theta \mathcal{L}_{PPO}(z_i, \theta_\tau)
$$

4.  **Influence Scoring**: An influence score is calculated as the dot product between the episode gradient and the validation gradient:

$$
\text{Score}(z_i) = g_{train} \cdot \bar{g}_{val}
$$

5.  **Filtering and Reweighting**: Episodes with $\text{Score}(z_i) \leq 0$ are discarded. Remaining positive episodes are assigned a normalized weight $w_i$ to maintain learning rate stability:

$$
w_i = \frac{\text{Score}(z_i)}{\frac{1}{|D'|} \sum_{z_j \in D'} \text{Score}(z_j)}
$$

6.  **Optimization**: The policy is updated using the refined, reweighted rollout buffer.

### Key Formulas
The total reward $r_{total}$ incorporates a KL divergence penalty to prevent the policy from deviating too far from the reference model $\pi_{ref}$:

$$
r_{total}(x, y) = R_\phi(x, y) - \beta \log \left(\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right)
$$

The PPO clipped surrogate loss $\mathcal{L}^{CLIP}$ used to derive $g_{train}$ is:

$$
\mathcal{L}^{CLIP}(\theta) = - \min \left(\rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right)
$$

where $\rho_t(\theta)$ is the probability ratio between the current and old policy.

### Key Quantitative Results
I-PPO was evaluated across five models (Rho-1B, Gemma-2-2B, Qwen2.5-3B, Phi-3-4B, LLaMA-3-8B) and five reasoning datasets (GSM8K, CollegeMath, MATH, OlympiadBench, ECQA).

*   **Performance Gains**: I-PPO consistently outperformed SFT and traditional PPO in Majority Vote (MV) and Exact Match (EM) accuracy. For example, on the Rho-1B model:
    *   **GSM8K**: I-PPO achieved 51.93% MV, compared to 50.05% for PPO and 43.52% for SFT.
    *   **MATH**: I-PPO achieved 23.80% MV, compared to 21.40% for PPO and 12.20% for SFT.
*   **Training Efficiency**: I-PPO acts as an intrinsic early stopping mechanism. As the model converges, the proportion of negative influence episodes increases, shrinking the rollout buffer and reducing the computational time per step in the latter half of training.
*   **Reasoning Quality**: Qualitative analysis and Sparse Autoencoder (SAE) interpretation revealed that I-PPO effectively filters "unfaithful" reasoning patterns, including "False Positives" (correct answer, flawed logic), "Nonsensical Reasoning," and "Reasoning Shortcuts."

### Stated Limitations
The primary limitation is the sensitivity of the data attribution process to the quality of the validation set. Noise or suboptimal data within the reference validation set $\mathcal{D}_{val}$ can degrade the accuracy of the influence score calculations.
