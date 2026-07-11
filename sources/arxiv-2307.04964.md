---
id: arxiv:2307.04964
type: paper
title: 'Secrets of RLHF in Large Language Models Part I: PPO'
url: https://arxiv.org/abs/2307.04964
retrieved: '2026-07-10'
maturity: comprehensive
topic: ppo-for-llms
---

**Core Problem**
Aligning large language models (LLMs) with human values—specifically the helpful, honest, and harmless (3H) framework—is essential for safe deployment. While supervised fine-tuning (SFT) mitigates some risks, it leaves models below human standards in safety and groundedness. Reinforcement learning from human feedback (RLHF) serves as the primary alignment paradigm, yet its implementation via Proximal Policy Optimization (PPO) is notoriously unstable. The authors identify that vanilla PPO suffers from sparse rewards, inefficient exploration in discrete word space, extreme hyperparameter sensitivity, and prohibitive trial-and-error costs. Coordinating four interdependent models (policy, value, reward, and reference) and relying on reward models trained on low-quality data frequently misguides policy optimization, creating a significant barrier to reliable LLM alignment.

**Methodological Framework**
The RLHF pipeline executes three sequential stages. First, an SFT model is trained to imitate human-annotated dialogues. Second, a reward model (RM) is trained on paired comparisons of preferred versus dispreferred responses, augmented with imitation learning to reinforce preferred token sequences. Third, the policy model is optimized via PPO using the RM as a reward signal. To address vanilla PPO instability, the authors propose **PPO-max**, an advanced variant that aggregates empirically effective low-level implementations and carefully calibrates them to prevent algorithmic interference. Training stability is monitored using action-space metrics—specifically perplexity, response length, and KL divergence relative to the SFT baseline—rather than relying solely on reward or loss values, which are less informative of training dynamics.

**Key Formulations**
The reward modeling loss combines a pairwise comparison term with an autoregressive language modeling loss on preferred responses:

$$
\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right] + \lambda \mathcal{L}_{LM}
$$

where $y_w$ and $y_l$ denote preferred and dispreferred responses, and $\mathcal{L}_{LM}$ applies imitation learning to the preferred sequence. During RL, the total reward incorporates a KL penalty to constrain policy deviation from the SFT model $\pi_{SFT}$:

$$
R_{total} = R_{RM} - \beta \cdot D_{KL}[\pi_\theta || \pi_{SFT}]
$$

Policy optimization employs Generalized Advantage Estimation (GAE) to balance bias and variance in advantage estimation:

$$
\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

The PPO-Clip objective restricts policy updates via a clipped surrogate:

$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$. The critic model minimizes mean squared error against estimated returns, and pretraining gradients may be mixed to preserve language capabilities:

$$
\mathcal{L}_{total} = \mathcal{L}^{PPO} + \alpha \mathcal{L}_{ptx}
$$

**Quantitative Results and Limitations**
The authors report evaluating PPO-max on 7B and 13B SFT models, claiming alignment performance comparable to ChatGPT and enhanced semantic comprehension. However, the provided excerpt concludes before presenting specific numerical benchmarks, reward scores, or comparative metrics. Stated limitations include the inherent instability of RLHF training, the critical dependency on reward model quality to avoid policy collapse, the computational burden of coordinating multiple models, and the sensitivity of PPO to hyperparameter configurations. The authors emphasize that without robust, code-level optimizations and careful constraint calibration, RLHF remains prohibitively expensive and unreliable for large-scale LLM alignment.
