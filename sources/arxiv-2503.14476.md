---
id: arxiv:2503.14476
type: paper
title: 'DAPO: An Open-Source LLM Reinforcement Learning System at Scale'
url: https://arxiv.org/abs/2503.14476
retrieved: '2026-07-10'
maturity: comprehensive
topic: grpo
---

**Core Problem**
Reproducing state-of-the-art reasoning LLMs trained with large-scale reinforcement learning is hindered by opaque training pipelines. Naive implementations of PPO and GRPO consistently fail to scale, suffering from entropy collapse (where policy diversity rapidly vanishes), reward noise from truncated generations, training instability, and vanishing gradients when prompts achieve 100% accuracy. These obstacles prevent the community from replicating industry-level results despite public technical reports.

**Method and Recipe**
The authors introduce DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization), a fully open-sourced RL system built on the `verl` framework. The training recipe begins by removing the KL divergence penalty, as long-chain-of-thought reasoning inherently diverges from the base model. The algorithm then applies four synergistic techniques. First, **Clip-Higher** decouples the upper and lower clipping bounds to prevent entropy collapse, allowing low-probability exploration tokens to increase their probability without being overly constrained. Second, **Dynamic Sampling** over-samples prompts and filters out those with 0% or 100% accuracy, ensuring every batch contains prompts with non-zero advantage signals and maintaining consistent gradient magnitudes. Third, the loss aggregation shifts from sample-level to **Token-Level Policy Gradient Loss**, averaging gradients per token before batch reduction to prevent unhealthy entropy/length growth and ensure long sequences contribute proportionally to updates. Fourth, **Overlong Reward Shaping** masks the loss for truncated samples and applies a soft, length-aware punishment to mitigate reward noise. The system trains on DAPO-Math-17K, a curated dataset of 17,000 math prompts with answers standardized to integers for reliable rule-based reward computation.

**Key Formulas**
The DAPO objective maximizes:
$$\max_{\theta} \mathbb{E}_{q, o_i} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t} \hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon_l, \epsilon_h) \hat{A}_{i,t} \right) \right]$$
where $r_{i,t}$ is the importance sampling ratio, $\hat{A}_{i,t}$ is the group-normalized advantage, and $\epsilon_l = 0.2$, $\epsilon_h = 0.28$. Dynamic Sampling filters samples where $r_i \in \{0, 1\}$. The token-level loss explicitly averages over $|o_i|$ before aggregation. Overlong punishment applies $R_{total} = R_{rule} - \lambda \cdot \max(0, |o_i| - L_{max} - \Delta)$, where $L_{max}=16,384$ and $\Delta=4,096$.

**Quantitative Results**
Evaluated on Qwen2.5-32B, DAPO achieves 50 on AIME 2024 (avg@32), outperforming DeepSeek-R1-Zero-Qwen-32B (47) while requiring only 50% of its training steps. Ablation studies demonstrate incremental gains: naive GRPO reaches 30; adding overlong filtering yields 36; Clip-Higher adds 2 points (38); soft overlong punishment adds 3 points (41); token-level loss adds 1 point (42); and dynamic sampling pushes the final score to 50. Training utilizes a batch size of 512, 16 rollouts per prompt, a constant learning rate of $10^{-6}$, and a maximum generation length of 20,480 tokens.

**Stated Limitations**
The authors note that final training rewards often correlate poorly with validation accuracy, indicating potential overfitting to the training distribution. Generated response lengths do not monotonically increase and may stagnate or decline during certain training phases. The RL system exhibits high sensitivity to hyperparameter and data variations, where minor adjustments can amplify into substantial outcome deviations due to complex subsystem interdependencies. Furthermore, the methodology is currently validated exclusively on mathematical tasks using a web-scraped, integer-transformed dataset, leaving broader domain generalization and parsing robustness unverified.
