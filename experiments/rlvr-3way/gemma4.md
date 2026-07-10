Reinforcement Learning with Verifiable Rewards (RLVR) is a paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs) by optimizing policies against objective, binary feedback (e.g., correctness of a mathematical answer). Unlike standard RLHF, which relies on learned reward models, RLVR leverages ground-truth verifiers to drive the discovery and reinforcement of correct reasoning trajectories [source:arxiv:2506.14245][source:arxiv:2509.24981].

## Core Mechanism and the "Capability Boundary" Debate

The fundamental objective of RLVR is to maximize the probability of generating a correct final answer $a_i$ given a prompt $q$. The reward signal is typically binary: $R(y_i) = \mathcal{I}_{\mathrm{Ans}}(a_i)$, where $\mathcal{I}$ is an indicator function [source:arxiv:2506.14245].

A central theoretical debate in RLVR is whether the process merely improves **sampling efficiency** (reweighting existing correct paths in the base model) or genuinely **extends the capability boundary** (discovering new reasoning paths). 

*   **The Sampling Hypothesis:** Suggests all correct paths exist in the base model, and RL increases their probability [source:arxiv:2506.14245].
*   **The Extension Hypothesis:** Argues RLVR implicitly incentivizes the model to develop new, correct reasoning strategies [source:arxiv:2506.14245].

Evidence from the CoT-Pass@K metric—which requires both the final answer and the intermediate Chain-of-Thought (CoT) to be correct—indicates a persistent performance gap between RLVR-tuned models and base models even at high $K$ values (up to 1024) on AIME 2024/2025 [source:arxiv:2506.14245]. This suggests that RLVR creates fundamentally higher-quality reasoning trajectories that were not present in the base model's distribution [source:arxiv:2506.14245].

## Optimization Frameworks

### Generalized Policy Iteration (GPI) and GRPO
Most RLVR implementations utilize Group Relative Policy Optimization (GRPO) to avoid the need for a separate value network. GRPO samples $G$ responses per prompt and computes a group-relative advantage:
\[
\hat{A}(y_i) = \frac{R(y_i) - \mu_{\mathbf{Y}}}{\sigma_{\mathbf{Y}}}, \quad \mu_{\mathbf{Y}} = \frac{1}{G} \sum_{j=1}^G R(y_j), \quad \sigma_{\mathbf{Y}} = \sqrt{\frac{1}{G} \sum_{j=1}^G (R(y_j) - \mu_{\mathbf{Y}})^2}
\]
The policy is then updated via the gradient $\nabla_\theta J(\theta) \approx \frac{1}{G} \sum_{i=1}^G \hat{A}(y_i) \nabla_\theta \log \pi_\theta(y_i \mid q)$ [source:arxiv:2506.14245].

### Random Policy Valuation (ROVER)
A significant disagreement exists regarding the necessity of the GPI loop. While most work builds on PPO/GRPO, ROVER argues that math reasoning is a finite-horizon MDP with deterministic, tree-structured transitions, making complex GPI unnecessarily cumbersome [source:arxiv:2509.24981]. 

ROVER proposes that optimal actions can be recovered directly from the Q-function of a fixed uniform policy $\pi_u$. It uses an intrinsic Q-parameterization based on LLM logits:
\[
Q(s_t, a_t) = \rho \big( \log \pi_\theta(a_t | s_t) - \log \pi_{\theta_{\text{old}}}(a_t | s_t) \big)
\]
Empirical results show ROVER outperforming GRPO and PPO, achieving an average pass@1 improvement of +8.2 on Qwen3-8B-Base across benchmarks like AIME24 and GPQA-diamond [source:arxiv:2509.24981].

## Exploration and the "Exploration Ceiling"

A recurring failure mode in RLVR is the "exploration ceiling," where models converge on narrow, over-optimized behaviors and fail to discover novel strategies [source:arxiv:2602.02555][source:arxiv:2506.14758].

### The KL Divergence Conflict
There is a fundamental tension regarding how to regularize the policy:
*   **Reverse KL (Standard):** Used in GRPO/PPO, this exhibit mode-seeking behavior, trapping the policy within the support region of the reference model and preventing the discovery of high-reward solutions outside that region [source:arxiv:2510.03865].
*   **Forward KL (RAPO):** RAPO proposes replacing reverse KL with forward KL and entropy maximization:
    $$\mathcal{J}_{\mathrm{FKL}}(\theta) = \mathbb{E}[r(x, y)] - \alpha \mathbb{D}_{\mathrm{KL}}(\pi_{\mathrm{ref}}||\pi_{\theta}) + \beta H(\pi_{\theta})$$
    This forces the policy to cover the reference distribution more broadly, allowing it to solve previously intractable problems (e.g., AIME 2025 Hard, where base models scored 0.000 but RAPO-7B reached 0.479) [source:arxiv:2510.03865].

### Noise and Entropy Strategies
To break the ceiling, different noise injection methods are proposed:

| Method | Mechanism | Core Argument | Result |
| :--- | :--- | :--- | :--- |
| **PSN-RLVR** [source:arxiv:2602.02555] | Gaussian noise in MLP/FFN layers | Token-level noise (temperature) is "unstructured jitter"; parameter noise ensures trajectory-level coherence [source:arxiv:2602.02555]. | Pass@256 on AIME 2024 increased from 62.8% to 65.5% (Qwen3-4B) [source:arxiv:2602.02555]. |
| **Entropy Shaping** [source:arxiv:2506.14758] | Advantage augmentation $\psi(\mathcal{H}_t) = \min(\alpha \mathcal{H}_t, \frac{|A_t|}{\kappa})$ | High token entropy correlates with pivotal logical tokens and self-correction [source:arxiv:2506.14758]. | AIME25 Pass@256 rose from 50.0 to 53.3 for Qwen2.5-Base+GRPO [source:arxiv:2506.14758]. |

## The Paradox of Spurious Rewards and Entropy

Recent analysis has challenged classical RL intuitions regarding reward noise and entropy [source:arxiv:2512.16912]. 

1.  **Spurious Rewards:** While classical RL suggests noise hinders exploitation, some RLVR experiments show that random rewards (Bernoulli(1/2)) can actually improve reasoning in stronger models (e.g., R1-Distill-Llama-8B), though they degrade weaker models (Qwen2.5-Math-1.5B) [source:arxiv:2512.16912].
2.  **Entropy Minimization:** Contrary to the view that entropy collapse is always bad [source:arxiv:2506.14758], some evidence suggests that entropy minimization can reinforce correct trajectories in specific regimes [source:arxiv:2512.16912].
3.  **Clipping Bias:** In GRPO, the clipped surrogate objective is often thought to provide a stabilizing signal. However, theoretical bounding shows the clipping correction is negligible relative to the raw signal ($\frac{\mathbb{E}[|N_{\mathrm{raw}}|]}{\mathbb{E}[|C_{\mathrm{tot}}^{+}|]} \approx 17.15$), implying clipping acts primarily as an implicit entropy regularizer rather than a primary gradient driver [source:arxiv:2512.16912].

## Current status and trajectory

RLVR is currently a **rising and rapidly evolving** technique, specifically within the context of "reasoning models" (e.g., DeepSeek-R1 style). While GRPO is becoming a default baseline for verifiable rewards [source:arxiv:2506.14245][source:arxiv:2602.02555], there is a clear trajectory toward **non-GPI methods** like ROVER [source:arxiv:2509.24981] and **advanced exploration** mechanisms like PSN [source:arxiv:2602.02555] and RAPO [source:arxiv:2510.03865]. The field is currently shifting from simple accuracy maximization toward managing the exploration-exploitation trade-off via entropy and parameter-space manipulation [source:arxiv:2506.14758][source:arxiv:2512.16912].

## Key takeaways

*   **Capability Expansion:** RLVR does not just reweight base model paths; it generates fundamentally new, higher-quality reasoning trajectories [source:arxiv:2506.14245].
*   **Algorithmic Divergence:** There is a conflict between the use of complex GPI (GRPO/PPO) and simpler Q-valuation methods (ROVER), with the latter claiming superior efficiency in tree-structured MDPs [source:arxiv:2509.24981].
*   **Exploration Bottlenecks:** Reverse KL divergence creates a "support region" trap; Forward KL (RAPO) and Parameter-Space Noise (PSN) are used to break this ceiling [source:arxiv:2510.03865][source:arxiv:2602.02555].
*   **Entropy's Dual Role:** While entropy collapse is generally viewed as a failure mode [source:arxiv:2506.14758], it may be beneficial in certain regimes for stronger models [source:arxiv:2512.16912].

## Related topics

- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Reward modeling for LLMs](reward-modeling.md)