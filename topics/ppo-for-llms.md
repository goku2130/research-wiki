---
title: PPO for LLM fine-tuning (RLHF)
maturity: developing
updated: '2026-07-10'
sources:
- arxiv:1707.06347
- arxiv:1706.03741
- arxiv:2009.01325
- arxiv:2203.02155
- arxiv:2305.18290
- arxiv:2307.04964
- arxiv:2401.06080
- cameronrwolfe:ppo-for-llms-a-guide-for-normal-people
open_questions:
- Can the KL penalty be entirely removed in all RLHF scenarios if the reward model
  is sufficiently robust, or is it fundamentally necessary for exploration in diverse
  token spaces?
- How can the coordination overhead of the four-model PPO architecture be reduced
  without sacrificing the stability provided by the value and reference models?
---

Proximal Policy Optimization (PPO) is a reinforcement learning algorithm used to align large language models (LLMs) with human preferences by maximizing a reward signal while constraining policy updates. It serves as the core optimization engine in the Reinforcement Learning from Human Feedback (RLHF) pipeline to ensure stable convergence in high-dimensional action spaces [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people][source:arxiv:2203.02155].

## Policy Gradient Basics
LLM alignment is formulated as a Markov Decision Process (MDP) where the initial state $s_1$ is the input prompt, and each generated token $a_t$ is a sequential action [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people]. The state updates autoregressively by appending the predicted tokens to the sequence [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people].

### Vanilla Policy Gradient (VPG)
The goal is to optimize a policy $\pi_\theta$ to maximize the expected cumulative reward $J(\theta) = \mathbb{E}_\tau [ \sum_{t=1}^{T} r_t ]$ [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people]. Using the log-derivative trick, the gradient is estimated as:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \Psi_t \right]
$$

where $\Psi_t$ is a signal representing the quality of the action [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people]. Direct application of VPG often results in high variance [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people].

### Variance Reduction and the Advantage Function
To reduce variance, $\Psi_t$ is replaced by the advantage function $A(s, a) = Q(s, a) - V(s)$, which measures how much better a specific action is compared to the average action in that state [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people]. In practice, Generalized Advantage Estimation (GAE) is used to balance bias and variance:

$$
\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

where $\gamma$ is the discount factor and $\lambda$ is a smoothing parameter [source:arxiv:2307.04964].

## Reward Modeling
Because human feedback is prohibitively expensive to collect for every RL step, a Reward Model (RM) is trained as a proxy for human judgment [source:arxiv:1706.03741][source:arxiv:2203.02155].

### The Bradley-Terry Model
RMs are typically trained on pairwise comparisons. Based on the Bradley-Terry model, the probability that a human prefers trajectory $\tau_1$ over $\tau_2$ is:

$$
P(\tau_1 \succ \tau_2) = \frac{e^{R(\tau_1)}}{e^{R(\tau_1)} + e^{R(\tau_2)}}
$$

The RM is trained by minimizing the negative log-likelihood (or cross-entropy loss) of the preferred summary relative to the rejected one:

$$
\mathcal{L}_{RM}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} [\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]
$$

where $y_w$ is the winning (preferred) response and $y_l$ is the losing response [source:arxiv:1706.03741][source:arxiv:2009.01325].

### Advanced RM Refinements
To address noise in human labels and out-of-distribution (OOD) instability, several enhancements are employed:
*   **Imitation Learning:** Some frameworks add an autoregressive language modeling loss $\mathcal{L}_{LM}$ on preferred responses to the RM objective [source:arxiv:2307.04964].
*   **Ensemble Denoising:** Using an ensemble of $N$ reward models to calculate preference strength $\mu = \frac{1}{N}\sum R_i$ and $\sigma = \text{std}(R_i)$ to partition and clean data [source:arxiv:2401.06080].
*   **Adaptive Margins:** Incorporating a margin in the loss function that scales with preference strength to enforce larger gaps for clear preferences [source:arxiv:2401.06080].
*   **Label Smoothing:** Modifying target distributions to prevent RM overconfidence [source:arxiv:2401.06080].

## PPO Mechanism and Clipping
Standard policy gradient methods are limited to one gradient update per data sample, which is sample-inefficient [source:arxiv:1707.06347]. PPO enables multiple epochs of minibatch updates on the same data by using a surrogate objective [source:arxiv:1707.06347].

### The Policy Ratio
PPO tracks the ratio between the current policy $\pi_\theta$ and the policy used to collect the data $\pi_{\theta_{old}}$:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

[source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people][source:arxiv:2307.04964].

### PPO-Clip Objective
To prevent "catastrophic" updates that move the policy too far from the trust region, PPO clips the policy ratio:

$$
\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

This ensures that the update is limited if the ratio $r_t(\theta)$ moves outside the interval $[1-\epsilon, 1+\epsilon]$ [source:arxiv:2307.04964].

## KL Penalty and Stability
Directly optimizing for the RM can lead to "reward hacking," where the model finds loopholes in the RM to achieve high scores without improving actual quality [source:arxiv:2009.01325].

### The KL Constraint
To prevent the policy from collapsing or drifting too far from the initial Supervised Fine-Tuning (SFT) model $\pi_{ref}$, a Kullback-Leibler (KL) divergence penalty is added to the reward:

$$
R(x, y) = r_\theta(x, y) - \beta \log \frac{\pi_\phi(y|x)}{\pi_{ref}(y|x)}
$$

where $\beta$ controls the strength of the penalty [source:arxiv:2009.01325][source:arxiv:2307.04964].

### Divergent Views on KL Necessity
There is a technical disagreement regarding the necessity of the KL penalty:
*   **Standard view:** The KL term is essential to act as an entropy bonus, encourage exploration, and prevent the policy from generating outputs outside the RM's training distribution [source:arxiv:2009.01325].
*   **Denoising view:** [source:arxiv:2401.06080] argues that if the reward model is sufficiently cleaned via ensemble-based denoising and adaptive margins, the KL penalty can be omitted without destabilizing training.

### Stability Mitigation
To combat the "alignment tax" (performance regression on general NLP tasks), PPO-ptx mixes PPO updates with the original pretraining log-likelihood maximization [source:arxiv:2203.02155]. Stability is monitored via action-space metrics—perplexity, response length, and KL divergence—rather than just reward scores [source:arxiv:2307.04964].

## Current status and trajectory
PPO has historically been the default algorithm for LLM alignment [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people]. However, it is currently facing competition from Direct Preference Optimization (DPO), which eliminates the need for a separate RM and the instability of RL by using a closed-form optimal policy derived from the RLHF objective [source:arxiv:2305.18290]. DPO is reported to exceed PPO in sentiment control and match it in summarization with significantly lower computational overhead [source:arxiv:2305.18290]. Despite this, PPO remains a primary tool for complex alignment (e.g., the 3H framework) where iterative refinement and dynamic reward signals are required [source:arxiv:2307.04964].

## Key takeaways
*   **PPO vs. VPG:** PPO improves sample efficiency and stability via a clipped surrogate objective, allowing multiple updates per batch [source:arxiv:1707.06347][source:arxiv:2307.04964].
*   **Reward Pipeline:** RLHF follows a strict sequence: SFT $\rightarrow$ Reward Modeling (via Bradley-Terry) $\rightarrow$ PPO Optimization [source:arxiv:2203.02155].
*   **Stability Tools:** GAE is used for advantage estimation, and KL penalties prevent reward hacking and policy collapse [source:arxiv:2009.01325][source:arxiv:2307.04964].
*   **Implementation Complexity:** PPO is computationally expensive and sensitive to hyperparameters, requiring the coordination of four models (policy, value, reward, and reference) [source:arxiv:2307.04964].

## Related topics
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)

## References
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [source:arxiv:1706.03741] [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
- [source:arxiv:2009.01325] [Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2307.04964] [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964)
- [source:arxiv:2401.06080] [Secrets of RLHF in Large Language Models Part II: Reward Modeling](https://arxiv.org/abs/2401.06080)
- [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people] [PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm)
