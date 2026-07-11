---
title: KL regularization in RLHF
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2203.02155
- arxiv:2305.18290
- arxiv:2407.13399
- arxiv:2502.01203
- arxiv:2411.04625
- arxiv:2601.16403
- rlhfbook:direct-alignment-algorithms
- iclr-blogposts:the-n-implementation-details-of-rlhf-wit
open_questions:
- 'Overoptimization**: Can stronger regularizers like $\chi^2$-divergence fully address
  reward overoptimization in large-scale RLHF, or is it an inherent limitation of
  offline alignment?'
- 'Reference Model Selection**: What are the best practices for selecting reference
  models (e.g., SFT vs. pretrained, single vs. multiple) to balance alignment performance
  and computational cost?'
- 'Hyperparameter Tuning**: Can automated methods (e.g., Bayesian optimization, meta-learning)
  reliably set $\beta$ and adaptive KL control parameters across diverse models and
  datasets?'
- 'Generalization**: How can theoretical guarantees for KL-regularized RLHF be extended
  to unbounded reward functions and infinite policy classes?'
---

# KL Regularization in RLHF: Reference Model, Per-Token KL, and the Boltzmann Optimum

KL regularization is the cornerstone of modern Reinforcement Learning from Human Feedback (RLHF), ensuring that fine-tuned language models remain anchored to their pretrained knowledge while optimizing for human preferences. This deep dive dissects the mathematical foundations, implementation nuances, and theoretical guarantees of KL regularization, focusing on the reference model, per-token penalties, and the optimal policy structure.

---

## Mathematical Foundations

### The RLHF Objective with KL Regularization
The standard RLHF objective maximizes an expected reward while constraining the policy $\pi_\phi$ to stay close to a reference model $\pi_{\text{ref}}$ (typically the supervised fine-tuned (SFT) model) via a KL divergence penalty:

$$
\mathcal{J}(\phi) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\phi(\cdot|x)} \left[ r_\theta(x, y) - \beta \cdot \text{KL}(\pi_\phi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) \right],
$$

where:
- $r_\theta(x, y)$ is the scalar reward predicted by a reward model for prompt $x$ and completion $y$,
- $\beta > 0$ is a hyperparameter controlling the strength of the KL penalty,
- $\text{KL}(\pi_\phi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\phi(\cdot|x)} \left[ \log \frac{\pi_\phi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$ is the reverse KL divergence.

This objective is derived from the constrained optimization framework where the KL penalty acts as a regularizer to prevent overfitting to the reward model and preserve the linguistic capabilities of the reference model [source:rlhfbook:direct-alignment-algorithms].

### The Boltzmann Optimum
The optimal policy $\pi^*$ that maximizes Eq. (1) can be derived in closed form using variational methods. Introducing a partition function $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(\beta^{-1} r_\theta(x, y))$, the optimal policy is:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r_\theta(x, y)\right).
$$

This is the *Boltzmann distribution* over completions, where the reward $r_\theta(x, y)$ acts as an energy term. The partition function $Z(x)$ ensures normalization. This result is foundational to both RLHF and Direct Preference Optimization (DPO) [source:arxiv:2305.18290][source:rlhfbook:direct-alignment-algorithms].

### Per-Token KL Penalty
In practice, the KL divergence in Eq. (1) is approximated by sampling a single trajectory $y \sim \pi_\phi(\cdot|x)$ and computing the per-token log-ratio sum:

$$
\text{KL}_{\text{sampled}} = \sum_{t=1}^T \log \frac{\pi_\phi(y_t|x, y_{<t})}{\pi_{\text{ref}}(y_t|x, y_{<t})}.
$$

This per-token penalty is critical for stabilizing training, as it prevents the policy from deviating too far from the reference model at any single step. During implementation, the penalty is scaled by $\beta$ and subtracted from the reward model score to form the adjusted reward [source:iclr-blogposts:the-n-implementation-details-of-rlhf-wit]:

$$
\tilde{r}(x, y) = r_\theta(x, y) - \beta \cdot \text{KL}_{\text{sampled}}.
$$

---

## Reference Model: Role and Variants

### Purpose of the Reference Model
The reference model $\pi_{\text{ref}}$ serves three primary functions:
1. **Regularization**: It anchors the policy to the pretrained or SFT model, preventing catastrophic forgetting of linguistic capabilities and mitigating reward hacking [source:arxiv:2203.02155].
2. **Reward Shaping**: In DPO, the reference model implicitly defines the reward function via the log-probability ratio $\log \frac{\pi_\phi(y|x)}{\pi_{\text{ref}}(y|x)}$ [source:arxiv:2305.18290].
3. **Initialization**: The reference model is typically used to initialize the policy $\pi_\phi$, ensuring a warm start for RL or DPO training.

### Single vs. Multiple Reference Models
#### Single Reference Model
The standard RLHF pipeline uses a single reference model, typically the SFT model. This is simple and effective but has limitations:
- **Bias Inheritance**: The reference model may inherit biases or limitations from its pretraining or SFT data, which are then propagated to the aligned policy.
- **Lack of Diversity**: A single reference model may not capture the full diversity of desirable behaviors, especially in multi-objective alignment (e.g., helpfulness vs. harmlessness) [source:arxiv:2502.01203].

#### Multiple Reference Models
Recent work extends RLHF to incorporate $K$ reference models $\{\pi_{\text{ref},i}\}_{i=1}^K$, each weighted by $\alpha_i$ or $\beta_i$ [source:arxiv:2502.01203]. The composite reference model is constructed as:
- **Reverse KL (RKL)**: $\widehat{\pi}_{\boldsymbol{\alpha},\text{ref}}(y|x) = \frac{\prod_{i=1}^K \pi_{\text{ref},i}^{\alpha_i}(y|x)}{F_{\boldsymbol{\alpha}}(x)}$, where $F_{\boldsymbol{\alpha}}(x)$ is a normalization constant.
- **Forward KL (FKL)**: $\bar{\pi}_{\boldsymbol{\beta},\text{ref}}(y|x) = \sum_{i=1}^K \beta_i \pi_{\text{ref},i}(y|x)$.

The optimal policy under RKL is:

$$
\pi_{\theta^*}^\gamma(y|x) = \frac{\widehat{\pi}_{\boldsymbol{\alpha},\text{ref}}(y|x)}{Z(x)} \exp(\gamma r_{\theta^*}(x,y)),
$$

where $\gamma$ is the inverse temperature. For FKL, the solution is implicit and requires solving a fixed-point equation.

**Empirical Impact**: Multi-reference RLHF improves alignment performance by leveraging diverse pretrained models. On UltraFeedback, multi-reference DPO achieves a win rate of 66.1%, compared to 56.4–59.8% for single-reference DPO [source:arxiv:2502.01203].

---

## Per-Token KL: Implementation and Nuances

### Computation in Practice
The per-token KL penalty is computed as follows during training:
1. **Forward Pass**: For a sampled completion $y \sim \pi_\phi(\cdot|x)$, compute the log-probabilities $\log \pi_\phi(y_t|x, y_{<t})$ and $\log \pi_{\text{ref}}(y_t|x, y_{<t})$ for each token $y_t$.
2. **KL Calculation**: Sum the per-token log-ratio differences:

$$
\text{KL}_{\text{sampled}} = \sum_{t=1}^T \left( \log \pi_\phi(y_t|x, y_{<t}) - \log \pi_{\text{ref}}(y_t|x, y_{<t}) \right).
$$

3. **Reward Adjustment**: Subtract $\beta \cdot \text{KL}_{\text{sampled}}$ from the reward $r_\theta(x, y)$ to form the adjusted reward:

$$
\tilde{r}(x, y) = r_\theta(x, y) - \beta \cdot \text{KL}_{\text{sampled}}.
$$

This adjusted reward is used in the PPO objective [source:iclr-blogposts:the-n-implementation-details-of-rlhf-wit].

### Adaptive KL Control
To prevent the policy from drifting too far from the reference model, adaptive KL control dynamically adjusts $\beta$ during training. The update rule is:

$$
\beta_{t+1} = \beta_t \cdot \left(1 + \text{proportional\_error} \cdot \frac{n_{\text{steps}}}{\text{horizon}}\right),
$$

where:
- $\text{proportional\_error} = \text{clip}\left(\frac{\text{current\_KL}}{\text{target\_KL}} - 1, -0.2, 0.2\right)$,
- $n_{\text{steps}}$ is the number of training steps,
- $\text{horizon}$ is a hyperparameter (e.g., 10,000 steps).

This ensures that the KL divergence remains close to a target value (e.g., 6 nats). Typical hyperparameters include $\text{init\_kl\_coef}=0.15$ and $\text{target}=6$ [source:iclr-blogposts:the-n-implementation-details-of-rlhf-wit].

---

## Theoretical Guarantees and Limitations

### Sample Complexity and Generalization
#### Single Reference Model
Under the Bradley-Terry preference model, the sample complexity of KL-regularized RLHF scales with the *single-policy concentrability coefficient* $\mathcal{C}^{\pi^*}$:

$$
\mathcal{C}^{\pi^*} = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi^*(\cdot|x)} \left[ \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} \right].
$$

This measures how well the reference model covers the optimal policy. The sub-optimality gap for DPO is bounded by terms involving $\mathcal{C}^{\pi^*}$ and the number of preference pairs [source:rlhfbook:direct-alignment-algorithms].

#### Multiple Reference Models
For multi-reference RLHF, the sample complexity depends on the coverage of the composite reference model. Under RKL, the sub-optimality gap converges at $O(1/n)$, while the optimality gap converges at $O(1/\sqrt{n})$. The bounds scale with $\gamma C_{\boldsymbol{\alpha},\varepsilon_{\text{rkl}}}$ and $R_{\max}$, the maximum reward [source:arxiv:2502.01203].

### Overoptimization and Pessimism
KL regularization is provably insufficient to prevent *reward overoptimization*, where the policy exploits inaccuracies in the reward model to achieve high rewards while degrading true quality. This occurs because KL divergence does not enforce pessimism in the face of uncertainty [source:arxiv:2407.13399].

#### $\chi^2$-Preference Optimization ($\chi_{\text{PO}}$)
To address this, $\chi_{\text{PO}}$ replaces KL regularization with $\chi^2$-divergence, which penalizes off-manifold behavior more aggressively. The objective is:

$$
\mathcal{L}_{\chi_{\text{PO}}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \text{clip}_{2R_{\max}} \left[ \beta \phi \left( \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} \right) - \beta \phi \left( \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right] \right) \right],
$$

where $\phi(z) = z + \log z$ and $\text{clip}_{2R_{\max}}$ bounds the density ratios. $\chi_{\text{PO}}$ achieves sample complexity scaling with $\mathcal{C}^{\pi^*}$ and is robust to overoptimization [source:arxiv:2407.13399].

### Disagreements in the Literature
1. **KL vs. $\chi^2$ Regularization**:
   - KL regularization is widely used and empirically effective, but [source:arxiv:2407.13399] argues it is fundamentally too weak to prevent overoptimization.
2. **Single vs. Multiple Reference Models**:
   - [source:arxiv:2502.01203] demonstrates empirical benefits of multi-reference models, but theoretical guarantees assume bounded rewards and finite reward classes, which may not hold in practice.
3. **Per-Token KL Implementation**:
   - [source:iclr-blogposts:the-n-implementation-details-of-rlhf-wit] shows that per-token KL computation is critical for stable training, but implementation details (e.g., adaptive KL control) vary across frameworks.

---

## Current Status and Trajectory

### Adoption and Trends
- **KL Regularization in RLHF**: KL regularization remains the *de facto* standard in RLHF pipelines, used in PPO, DPO, and their variants. It is the default in major alignment frameworks (e.g., HuggingFace TRL, DeepSpeed RLHF) and deployed in models like Llama 3 Instruct and Zephyr [source:rlhfbook:direct-alignment-algorithms].
- **DPO and Direct Alignment**: DPO has rapidly gained traction due to its simplicity and competitive performance. It is now the default for offline preference optimization, though PPO remains dominant in online settings [source:arxiv:2305.18290].
- **Multi-Reference RLHF**: Early adoption is reported in research settings (e.g., Qwen 2.5 7B), but not widely deployed in production systems. The technique is rising but not yet default [source:arxiv:2502.01203].
- **$\chi^2$-Preference Optimization**: Proposed as a theoretical improvement, but not widely reported in large-scale deployments. Its trajectory depends on empirical validation at scale.

### Challenges and Open Problems
1. **Overoptimization**: KL regularization does not fully address reward overoptimization. Stronger regularizers (e.g., $\chi^2$-divergence) and uncertainty-aware reward modeling are active areas of research.
2. **Reference Model Selection**: The choice of reference model (SFT vs. pretrained, single vs. multiple) significantly impacts alignment performance, but best practices are not yet established.
3. **Hyperparameter Sensitivity**: $\beta$ and adaptive KL control parameters are highly sensitive to the base model and dataset, requiring extensive tuning.
4. **Generalization**: Theoretical guarantees assume bounded rewards and finite policy classes, which may not hold in practice.

### Trajectory
- **KL Regularization**: Likely to remain the default for the foreseeable future, given its simplicity and effectiveness. However, its limitations may drive adoption of stronger regularizers like $\chi^2$-divergence.
- **DPO**: Continues to rise as the preferred method for offline alignment, with ongoing work to address its limitations (e.g., preference displacement, offline data constraints).
- **Multi-Reference RLHF**: Early signs of adoption in research, but broader deployment depends on scalability and tooling improvements.
- **$\chi_{\text{PO}}$**: Theoretical promise but unproven at scale. If empirical results validate its superiority, it could displace KL regularization in future pipelines.

---

## Key Takeaways
- **Boltzmann Optimum**: The optimal policy under KL-regularized RLHF is a Boltzmann distribution over completions, where the reward acts as an energy term.
- **Per-Token KL**: Critical for stable training, computed as the sum of log-probability ratios between the policy and reference model. Adaptive KL control dynamically adjusts $\beta$ to maintain a target KL divergence.
- **Reference Model**: Anchors the policy to pretrained knowledge, with single-reference models being the default. Multi-reference models improve alignment performance but are not yet widely deployed.
- **Theoretical Guarantees**: Sample complexity scales with the single-policy concentrability coefficient $\mathcal{C}^{\pi^*}$. KL regularization is insufficient to prevent overoptimization; $\chi^2$-divergence is proposed as a stronger alternative.
- **Current Status**: KL regularization and DPO are the defaults for RLHF. Multi-reference methods and $\chi_{\text{PO}}$ are rising but not yet widely adopted.

---

## Related Topics
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)

---

##

## References
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2407.13399] [Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization via Chi-Squared Preference Optimization](https://arxiv.org/abs/2407.13399)
- [source:arxiv:2502.01203] [KL-Regularized RLHF with Multiple Reference Models: Exact Solutions and Sample Complexity](https://arxiv.org/abs/2502.01203)
- [source:arxiv:2411.04625] [Sharp Analysis for KL-Regularized Contextual Bandits and RLHF](https://arxiv.org/abs/2411.04625)
- [source:arxiv:2601.16403] [Towards a Theoretical Understanding to the Generalization of RLHF](https://arxiv.org/abs/2601.16403)
- [source:rlhfbook:direct-alignment-algorithms] [Direct-Alignment Algorithms](https://rlhfbook.com/c/08-direct-alignment)
- [source:iclr-blogposts:the-n-implementation-details-of-rlhf-wit] [The N Implementation Details of RLHF with PPO](https://iclr-blogposts.github.io/2024/blog/the-n-implementation-details-of-rlhf-with-ppo/)
