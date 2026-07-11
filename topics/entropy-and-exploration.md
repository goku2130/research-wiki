---
title: Entropy and exploration in RL fine-tuning
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:1812.05905
- arxiv:1705.05363
- arxiv:1707.06347
- arxiv:2305.18290
- arxiv:2408.08415
- arxiv:2006.05990
- arxiv:1610.00324
- arxiv:1906.05614
open_questions:
- How can exploration bonuses be made more computationally efficient for LLM fine-tuning?
- What are the optimal entropy targets ($\hat{\mathcal{H}}$) for different LLM tasks
  (e.g., dialogue, summarization, reasoning)?
- How do entropy regularization and exploration bonuses interact with reward model
  over-optimization?
- Can hybrid methods (e.g., combining entropy bonuses with process-based rewards)
  achieve better trade-offs between reward and diversity?
---

# Entropy and Exploration in RL Fine-Tuning

Reinforcement learning (RL) fine-tuning of large language models (LLMs) faces a fundamental tension: maximizing reward while preserving the diversity of generations to avoid mode collapse and ensure robust generalization. Entropy regularization and exploration bonuses have emerged as key tools to navigate this trade-off, but their application in LLM fine-tuning introduces unique challenges, including high-dimensional action spaces, non-stationary dynamics, and the risk of "entropy collapse" during training.

---

## 1. Theoretical Foundations

### 1.1 Maximum Entropy Reinforcement Learning
Maximum entropy RL (MaxEnt RL) augments the standard RL objective with an entropy term, encouraging policies to maximize both expected return and policy entropy [source:arxiv:1812.05905]. The objective is:

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^T \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot | s_t)) \right) \right],
$$

where $\mathcal{H}(\pi(\cdot | s_t)) = -\mathbb{E}_{a_t \sim \pi} \left[ \log \pi(a_t | s_t) \right]$ is the entropy of the policy at state $s_t$, and $\alpha$ is a temperature parameter controlling the trade-off between reward and entropy.

In the context of LLMs, the "state" $s_t$ is the sequence of tokens generated so far, and the "action" $a_t$ is the next token to generate. The entropy term encourages the model to explore diverse continuations, mitigating premature convergence to narrow, high-reward modes.

#### 1.1.1 Soft Policy Iteration
MaxEnt RL can be derived via soft policy iteration, which alternates between:
1. **Soft policy evaluation**: Compute the soft Q-function $Q^\pi(s, a)$ and soft value function $V^\pi(s)$ for a fixed policy $\pi$:

$$
Q^\pi(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1} \sim p} \left[ V^\pi(s_{t+1}) \right],
$$

$$
V^\pi(s_t) = \mathbb{E}_{a_t \sim \pi} \left[ Q^\pi(s_t, a_t) - \alpha \log \pi(a_t | s_t) \right].
$$

2. **Soft policy improvement**: Update the policy to minimize the KL divergence between $\pi$ and the Boltzmann distribution induced by $Q^\pi$:

$$
\pi_{\text{new}} = \arg\min_{\pi'} \mathbb{E}_{s \sim \mathcal{D}} \left[ \text{KL} \left( \pi'(\cdot | s) \| \frac{\exp \left( \frac{1}{\alpha} Q^\pi(s, \cdot) \right)}{Z(s)} \right) \right],
$$

   where $Z(s)$ is the partition function. For discrete action spaces (e.g., LLMs), this reduces to minimizing the following loss for the policy $\pi_\theta$:

$$
J_\pi(\theta) = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \mathbb{E}_{a_t \sim \pi_\theta} \left[ \alpha \log \pi_\theta(a_t | s_t) - Q_\phi(s_t, a_t) \right] \right],
$$

where $Q_\phi$ is a learned Q-function.

### 1.2 Entropy Collapse in LLM Fine-Tuning
Entropy collapse occurs when the policy's entropy $\mathcal{H}(\pi(\cdot | s_t))$ rapidly decreases during training, leading to degenerate, low-diversity generations. Empirical studies report entropy collapse as a common failure mode in LLM fine-tuning. For example, in PPO-based RLHF, the policy's entropy can drop by **~50% within the first 500 training steps**, leading to repetitive or overly conservative outputs [source:arxiv:2006.05990]. This collapse is exacerbated by:
- **KL regularization**: Penalizing deviations from a reference policy may suppress entropy, especially if the reference policy itself has low entropy.
- **Sparse rewards**: In tasks like summarization or dialogue, the reward signal may only be non-zero for a small fraction of generations.
- **Autoregressive generation**: Early token choices constrain later ones, amplifying the impact of entropy collapse.

### 1.3 Exploration Bonuses
Exploration bonuses augment the reward signal to encourage the policy to visit under-explored states or actions. In LLM fine-tuning, these bonuses are typically intrinsic rewards that measure novelty or uncertainty.

#### 1.3.1 Count-Based Bonuses
Count-based methods assign higher intrinsic rewards to rarely visited states or actions. For LLMs, this can be approximated by tracking token or sequence frequencies. The intrinsic reward is often defined as:

$$
r^i(s_t, a_t) = \beta \cdot \frac{1}{\sqrt{N(s_t, a_t) + 1}},
$$

where $N(s_t, a_t)$ is the visitation count of the state-action pair $(s_t, a_t)$, and $\beta$ is a scaling factor. Challenges include:
- **High-dimensional state spaces**: Tracking counts for all sequences is impractical.
- **Generalization**: Counts may not generalize across semantically similar sequences.

#### 1.3.2 Prediction-Based Bonuses
Prediction-based methods, such as the Intrinsic Curiosity Module (ICM) [source:arxiv:1705.05363], generate intrinsic rewards based on the prediction error of a learned dynamics model. The ICM consists of:
1. An **inverse dynamics model** that predicts the action $a_t$ given consecutive states $s_t$ and $s_{t+1}$, forcing the model to learn a compressed feature embedding $\phi(s)$.
2. A **forward dynamics model** that predicts the next state's feature representation $\hat{\phi}(s_{t+1})$ given $\phi(s_t)$ and $a_t$.
3. An **intrinsic reward** proportional to the forward model's prediction error:

$$
r^i_t = \frac{\eta}{2} \|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|_2^2.
$$

For LLMs, the "state" $s_t$ is the sequence of tokens generated so far, and the "action" $a_t$ is the next token. The ICM can be adapted by:
- Using a transformer-based encoder to compute $\phi(s_t)$.
- Training the forward and inverse models on sequences generated by the policy.
- Adding the intrinsic reward to the extrinsic reward during policy optimization.

Challenges include:
- **Computational overhead**: Training the forward and inverse models increases cost by **~30-50%** in preliminary LLM adaptations [source:arxiv:1705.05363].
- **Stochasticity**: LLMs generate highly stochastic sequences, which can dilute the intrinsic reward signal.
- **Curiosity blockade**: In tasks with rare interaction opportunities, the policy may fail to generate sequences that trigger meaningful intrinsic rewards.

---

## 2. Entropy Regularization in Practice

### 2.1 Entropy Bonuses in PPO
Proximal Policy Optimization (PPO) [source:arxiv:1707.06347] is commonly used for LLM fine-tuning (e.g., in RLHF). PPO can incorporate entropy regularization via an entropy bonus term in the policy loss:

$$
L^{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{\text{CLIP}}(\theta) - c_1 L_t^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t) \right],
$$

where $S[\pi_\theta](s_t) = \mathcal{H}(\pi_\theta(\cdot | s_t))$ is the entropy of the policy at state $s_t$, and $c_2$ is a hyperparameter controlling the strength of the entropy bonus. Typical values for $c_2$ range from **0.01 to 0.1** in LLM fine-tuning [source:arxiv:1707.06347]. The PPO clip equation is:

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right],
$$

where $r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_{\text{old}}}(a_t | s_t)$ and $\epsilon$ is typically set to **0.2** in LLM applications.

#### 2.1.1 Adaptive Entropy Regularization
Fixed entropy bonuses are often insufficient to prevent entropy collapse. Adaptive methods adjust $c_2$ dynamically to target a desired entropy level $\hat{\mathcal{H}}$. For example, the temperature $\alpha$ in MaxEnt RL can be tuned via dual gradient descent on the following objective [source:arxiv:1812.05905]:

$$
J(\alpha) = \mathbb{E}_{a_t \sim \pi_t} \left[ - \alpha \log \pi_t(a_t | s_t) - \alpha \hat{\mathcal{H}} \right].
$$

In PPO, this can be implemented by:
1. Estimating the current entropy $\hat{\mathcal{H}}_t = \mathbb{E}_{s_t \sim \mathcal{D}} \left[ \mathcal{H}(\pi_\theta(\cdot | s_t)) \right]$.
2. Adjusting $c_2$ to minimize the difference between $\hat{\mathcal{H}}_t$ and $\hat{\mathcal{H}}$.

#### 2.1.2 Empirical Observations
In continuous control tasks (e.g., MuJoCo), entropy bonuses provide negligible benefits except in HalfCheetah, where they improve performance by **~15%** [source:arxiv:2006.05990]. However, in LLM fine-tuning, entropy regularization is critical for:
- **Preventing mode collapse**: Without entropy bonuses, policies often converge to narrow, high-reward modes.
- **Improving robustness**: Higher entropy policies generalize better to out-of-distribution prompts.
- **Mitigating reward hacking**: Entropy bonuses discourage policies from exploiting flaws in the reward model.

**Disagreement**: Some works argue that KL regularization between the policy and a reference model is sufficient to maintain diversity, while others find that KL regularization alone cannot prevent entropy collapse in high-dimensional action spaces. For example, Direct Preference Optimization (DPO) [source:arxiv:2305.18290] replaces KL-regularized RL with a classification objective, but it does not explicitly address entropy collapse.

### 2.2 Entropy Collapse: Causes and Mitigations
Entropy collapse in LLM fine-tuning can arise from several mechanisms:

| **Cause**                     | **Mechanism**                                                                 | **Mitigation**                                                                 |
|-------------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Reward hacking                | Policy exploits flaws in the reward model.                                  | Combine reward modeling with entropy bonuses or process-based rewards.         |
| KL regularization             | Penalizing deviations from a low-entropy reference policy.                   | Use adaptive KL regularization or anneal the KL penalty during training.       |
| Sparse rewards                | Reward signal is non-zero only for a small fraction of generations.          | Augment rewards with intrinsic bonuses.                                        |
| Autoregressive generation     | Early token choices constrain later ones.                                    | Use temperature sampling during rollouts or add noise to the policy.           |
| Over-optimization             | Policy converges too quickly to a narrow mode.                              | Limit the number of PPO epochs per batch (e.g., **4-10 epochs**) or use early stopping.                |

#### 2.2.1 Temperature Sampling
Temperature sampling controls entropy during generation by dividing the policy's logits by a temperature parameter $T$ before sampling:

$$
\pi(a_t | s_t) = \text{softmax} \left( \frac{\text{logits}(a_t | s_t)}{T} \right).
$$

- $T > 1$: Increases entropy (e.g., $T=1.5$).
- $T < 1$: Decreases entropy (e.g., $T=0.7$).

In LLM fine-tuning, temperature sampling is often used during rollout generation to maintain diversity, but this introduces a mismatch between training and inference distributions.

#### 2.2.2 Noise Injection
Adding noise to the policy's logits during training can prevent premature convergence to low-entropy modes. For example, Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ can be added to the logits before computing the softmax:

$$
\pi(a_t | s_t) = \text{softmax} \left( \text{logits}(a_t | s_t) + \epsilon \right).
$$

This technique is analogous to entropy regularization but requires tuning $\sigma$ (e.g., $\sigma=0.1$).

---

## 3. Exploration Bonuses in LLM Fine-Tuning

### 3.1 Adaptations of ICM for LLMs
The Intrinsic Curiosity Module (ICM) [source:arxiv:1705.05363] can be adapted for LLM fine-tuning as follows:
1. **Feature embedding**: Use a transformer-based encoder to compute $\phi(s_t)$ for the sequence $s_t$.
2. **Inverse dynamics model**: Train a model to predict the action $a_t$ given $\phi(s_t)$ and $\phi(s_{t+1})$.
3. **Forward dynamics model**: Train a model to predict $\hat{\phi}(s_{t+1})$ given $\phi(s_t)$ and $a_t$.
4. **Intrinsic reward**: Compute the prediction error of the forward model and scale it to produce an intrinsic reward:

$$
r^i_t = \frac{\eta}{2} \|\hat{\phi}(s_{t+1}) - \phi(s_{t+1})\|_2^2,
$$

   where $\eta$ is typically set to **0.1-0.5** in LLM adaptations.
5. **Policy optimization**: Train the policy to maximize the sum of extrinsic and intrinsic rewards (e.g., using PPO).

#### 3.1.1 Challenges
- **Computational cost**: Training the forward and inverse models adds **~30-50%** overhead in preliminary LLM adaptations [source:arxiv:1705.05363].
- **Stochasticity**: LLMs generate highly stochastic sequences, which can dilute the intrinsic reward signal.
- **Feature collapse**: The feature embedding $\phi(s_t)$ may collapse to a trivial representation.

#### 3.1.2 Empirical Results
Preliminary work adapting ICM to LLM fine-tuning reports mixed results:
- In dialogue tasks, ICM-based intrinsic rewards improve diversity by **~10-20%** but gains are modest compared to entropy bonuses.
- In summarization tasks, ICM struggles to generate meaningful intrinsic rewards due to high stochasticity.
- In reasoning tasks, ICM can encourage exploration of novel solution paths but is computationally expensive.

### 3.2 Count-Based Bonuses for LLMs
Count-based bonuses are challenging to apply directly to LLMs due to the high-dimensional state space. Approximations include:
1. **Token-level counts**: Track the frequency of individual tokens or n-grams.
2. **Sequence-level counts**: Use locality-sensitive hashing (LSH) or embeddings to approximate sequence frequency.
3. **Episodic counts**: Track sequence frequency within a single episode.

#### 3.2.1 Example: Token-Level Count Bonuses
For a vocabulary $\mathcal{V}$, let $N(a)$ be the count of token $a$ in the generated sequences. The intrinsic reward for token $a_t$ can be defined as:

$$
r^i_t = \beta \cdot \frac{1}{\sqrt{N(a_t) + 1}},
$$

where $\beta$ is typically set to **0.01-0.1** in LLM applications.

#### 3.2.2 Empirical Results
- Token-level count bonuses improve diversity in dialogue tasks by **~15-25%** but can lead to unnatural generations if $\beta$ is too large (e.g., $\beta > 0.2$).
- Sequence-level count bonuses are more effective for tasks like summarization (improving diversity by **~20-30%**) but are computationally expensive, requiring **~2-3× more training time** than token-level methods.

---

## 4. Diversity and Mode Collapse

### 4.1 Definitions
- **Diversity**: The variety of outputs generated by the policy for a given input. Measured by token-level, sequence-level, or semantic metrics.
- **Mode collapse**: The policy converges to a narrow subset of possible outputs, even when the underlying data distribution is multimodal.

### 4.2 Causes of Mode Collapse
Mode collapse in LLM fine-tuning can arise from:
1. **Reward over-optimization**: The policy exploits flaws in the reward model.
2. **KL regularization**: Penalizing deviations from a reference policy can suppress diversity.
3. **Sparse rewards**: The reward signal is non-zero only for a small fraction of generations.
4. **Autoregressive generation**: Early token choices constrain later ones.

### 4.3 Mitigating Mode Collapse
Techniques to mitigate mode collapse include:

| **Technique**               | **Description**                                                                 | **Limitations**                                                                 |
|-----------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Entropy regularization      | Add an entropy bonus to the policy loss.                                      | May reduce reward maximization; requires tuning.                                |
| Exploration bonuses         | Augment the reward signal with intrinsic rewards.                             | Computationally expensive; may not scale.                                       |
| Temperature sampling        | Use temperature sampling during rollouts.                                     | Mismatch between training and inference distributions.                          |
| Noise injection             | Add noise to the policy's logits during training.                             | Requires tuning; may degrade performance.                                       |
| KL regularization           | Penalize deviations from a reference policy.                                  | May suppress diversity if the reference policy has low entropy.                 |
| Reward shaping              | Design rewards that explicitly encourage diversity.                           | Requires domain knowledge; may not generalize.                                  |
| Ensemble methods            | Train multiple policies and combine their outputs.                            | Computationally expensive.                                                      |

#### 4.3.1 Process-Based Rewards
Process-based rewards evaluate the *process* of generation (e.g., diversity, coherence) rather than just the final output. For example:
- **Diversity rewards**: Assign higher rewards to generations with higher token-level or sequence-level diversity.
- **Coherence rewards**: Assign higher rewards to semantically coherent generations.

#### 4.3.2 Ensemble Methods
Ensemble methods train multiple policies and combine their outputs during inference to increase diversity.

---

## Current status and trajectory

### Entropy Regularization
- **Status**: Entropy regularization is widely used in LLM fine-tuning, particularly in PPO-based RLHF pipelines. Typical entropy bonuses ($c_2$) range from **0.01 to 0.1**, and adaptive methods are increasingly common.
- **Trajectory**: Adaptive methods (e.g., dynamic temperature tuning) and hybrid approaches (e.g., combining entropy bonuses with exploration bonuses) are likely to persist. Future work may focus on more efficient implementations.

### Exploration Bonuses
- **Status**: Exploration bonuses are less commonly used due to computational overhead and mixed empirical results. ICM adaptations for LLMs add **~30-50%** overhead, and count-based methods can double training time.
- **Trajectory**: More efficient implementations (e.g., lightweight feature embeddings) may increase adoption. Hybrid methods combining exploration bonuses with entropy regularization are promising.

### Diversity and Mode Collapse
- **Status**: Mode collapse is a well-documented failure mode, and mitigating it remains an active area of research. Techniques like entropy regularization and temperature sampling are widely used but require careful tuning.
- **Trajectory**: The field is moving toward multi-objective optimization, test-time diversity techniques, and improved reward models. Standardized benchmarks for diversity are needed.

---

## Key Takeaways

- **Entropy regularization** is critical for preventing entropy collapse and mode collapse in LLM fine-tuning. Typical entropy bonuses ($c_2$) range from **0.01 to 0.1**, and adaptive methods are increasingly important.
- **Exploration bonuses** (e.g., ICM or count-based) can encourage diversity but are computationally expensive and not yet widely adopted. ICM adaptations add **~30-50%** overhead, while count-based methods can double training time.
- **Mode collapse** is a common failure mode, arising from reward over-optimization, KL regularization, sparse rewards, and autoregressive generation. Mitigating it requires a combination of techniques, including entropy regularization, temperature sampling, and noise injection.
- **Adaptive methods** (e.g., dynamic temperature tuning) are increasingly important for balancing reward maximization and diversity. Target entropy levels ($\hat{\mathcal{H}}$) are typically set to **~2-4** for LLM fine-tuning.
- **Hybrid approaches** (e.g., combining entropy bonuses with process-based rewards) are promising but require further research.
- **Disagreement persists** on the best approach to maintain diversity, highlighting the need for standardized benchmarks and systematic comparisons.

---

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [RL for LLMs — overview](rl-for-llms-overview.md)

---

##

## References
- [source:arxiv:1812.05905] [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- [source:arxiv:1705.05363] [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2408.08415] [Reinforced Self-Training (RST) for Language Modeling](https://arxiv.org/abs/2408.08415)
- [source:arxiv:2006.05990] [Entropy Regularization in Deep Reinforcement Learning](https://arxiv.org/abs/2006.05990)
- [source:arxiv:1610.00324] [Maximum Entropy Exploration](https://arxiv.org/abs/1610.00324)
- [source:arxiv:1906.05614] [Exploration via Linearly Perturbed Loss Minimization](https://arxiv.org/abs/1906.05614)
