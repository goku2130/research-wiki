---
id: arxiv:2310.17022
type: paper
title: Controlled Decoding from Language Models
url: https://arxiv.org/abs/2310.17022
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# Controlled Decoding from Language Models

### Core Problem
The authors address the challenge of aligning generative language model (LM) responses to specific rewards when the pre-trained representations of the base model are frozen. Existing "generator improvement" methods (e.g., PPO, DPO) require updating model weights, which limits reward configurability at inference time. Conversely, "inference-time add-on" solutions like best-of-$K$ sampling suffer from high latency and inefficiency. The goal is to create a modular, inference-time alignment framework that provides a favorable trade-off between reward, KL divergence from the base model, and computational efficiency.

### Method: Controlled Decoding (CD)
Controlled Decoding (CD) solves a tokenwise KL-regularized reinforcement learning (RL) objective by employing a separate **prefix scorer** module. This module learns a value function $V_\theta$ that steers the frozen base model $\pi_{\text{ref}}$ toward high-reward outcomes.

#### 1. Training the Prefix Scorer
The authors propose two training recipes for the prefix scorer:
*   **CD-FUDGE:** A supervised approach where the scorer is trained on rollouts from the base model. The loss $\ell_F$ minimizes the squared difference between the prefix score and the final reward $r$ of the completed sequence:

$$
\ell_{F}(\mathbf{x},\mathbf{y};\theta)=\frac{1}{2}\sum_{t\in[|\mathbf{y}|]}\left(V_{\theta}([\mathbf{x},y^{t}])-r([\mathbf{x},\mathbf{y}])\right)^{2}
$$

*   **CD-Q:** An off-policy solver inspired by DQN that uses a Bellman identity. It minimizes the difference between the current value and a target $v_t$:

$$
\ell_{Q}(\mathbf{x},y^{t};\theta)=\frac{1}{2}\sum_{t\in[|\mathbf{y}|]}\big(V_{\theta}([\mathbf{x},y^{t}])-\dot{v}_{t}\big)^{2}
$$

    where $v_t$ is the expected value of the next token $z$ sampled from $\pi_{\text{ref}}$ if $y_t \neq EOS$, and the reward $r$ if $y_t = EOS$.

#### 2. Inference-Time Sampling Strategies
*   **Tokenwise Sampling:** The base model's logits are linearly combined with the prefix scorer's values to sample the next token $z$:

$$
\pi_{\theta}(z|[\mathbf{x},y^{t}]) \propto \pi_{\text{ref}}(z|[\mathbf{x},y^{t}])e^{\lambda V_{\theta}([\mathbf{x},y^{t},z])}
$$

*   **Blockwise Best-of-$K$:** The model samples $K$ independent continuation blocks of length $M$ from $\pi_{\text{ref}}$. The block with the highest prefix score $V_\theta$ is selected, and the process repeats until an $EOS$ token is accepted.

### Key Formulas
The framework is grounded in a KL-regularized RL objective $J_\lambda$ that trades off the advantage function $A$ (expected reward gain) against the tokenwise KL divergence $D$:

$$
J_{\lambda}([ \mathbf {x}, y ^ {t} ]; \pi) := \lambda A ([ \mathbf {x}, y ^ {t} ]; \pi) - D ([ \mathbf {x}, y ^ {t} ]; \pi)
$$

Theorem 2.1 proves that the unique optimal policy $\pi_\lambda^\star$ for this objective is:

$$
\pi_{\lambda}^{\star}(z|[\mathbf{x},y^{t}]) \propto p(z|[\mathbf{x},y^{t}])e^{\lambda V^{\star}([\mathbf{x},y^{t},z])}
$$

### Quantitative Results
*   **Efficiency vs. Best-of-$K$:** In response length experiments, blockwise CD-Q achieved similar reward-KL trade-offs as best-of-$K$ but with significantly fewer samples. For example, CD-Q with $K=6$ performed similarly to best-of-$K$ with $K=50$.
*   **Alignment Performance:** Blockwise CD-Q and CD-FUDGE substantially outperformed RL baselines like IPO and PPO in helpfulness and harmlessness (HH) win rates.
*   **Generalization:** Prefix scorers trained via CD-Q transferred to unseen base models (e.g., transferring from PaLM 2-XXS to PaLM 2-S or PaLM 2-XS) without retraining, maintaining performance on par with best-of-$K$.
*   **Hybrid Approach:** Combining DPO with blockwise CD-Q improved efficiency; for a KL value of 5, the hybrid approach required $K=8$ to match the performance of vanilla blockwise CD-Q at $K=32$.

### Limitations
*   **Restrictiveness:** The tokenwise RL formulation is more restrictive than the sequence-level RL used in standard RLHF/DPO.
*   **Scorer Noise:** Prefix scorers exhibited lower classification accuracy ($\approx 0.6$) compared to the reward models ($\approx 0.7$), which the authors attribute to noisy training data.
*   **Optimality:** Blockwise CD is not designed to optimally solve the sequence-level KL-regularized objective, though it performs well empirically.
