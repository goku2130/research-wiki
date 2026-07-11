---
title: PPO for LLM fine-tuning (RLHF)
maturity: comprehensive
updated: '2026-07-11'
sources:
- arxiv:2009.01325
- arxiv:2403.19279
- arxiv:2310.00212
- arxiv:2307.04964
- arxiv:1909.08593
- arxiv:1706.03741
- arxiv:1707.06347
- arxiv:2204.05862
- arxiv:2203.02155
- spinningup:spinning-up-proximal-policy-optimization
- cameronrwolfe:proximal-policy-optimization-ppo-the-key
open_questions:
- What is the optimal schedule for the KL penalty coefficient $\beta$ during PPO training,
  and does adaptive KL control outperform fixed coefficients?
- Can PPO be effectively combined with offline preference data (e.g., via hybrid online/offline
  objectives) to reduce sample complexity while retaining online benefits?
- How does the choice of advantage estimator (GAE vs. REINFORCE with baseline vs.
  learned critic) affect credit assignment in long-horizon generation tasks like tool
  use or multi-step reasoning?
- Is there a principled way to set the clipping threshold $\epsilon$ for LLMs, or
  does it require per-task tuning?
---

Proximal Policy Optimization (PPO) is the dominant reinforcement learning algorithm for aligning large language models via RLHF, combining a clipped surrogate objective with a KL penalty to stabilize policy updates against a learned reward model. This article provides a rigorous technical synthesis of the policy-gradient foundations, reward model integration, KL regularization mechanics, clipping behavior, and practical modifications that distinguish LLM PPO from its original continuous-control formulation.

## Policy-gradient foundations for LLM fine-tuning

In the LLM RLHF setting, the trajectory is a generated response $y$ conditioned on a prompt $x$, the action space is the vocabulary at each token position, and the reward is sparse — typically a single scalar $r(x,y)$ from a reward model assigned at sequence end [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key]. This reduces the problem to a contextual bandit: the advantage collapses to $r(x,y) - V(x)$ where $V(x)$ is a learned value function (critic) estimating the expected reward for prompt $x$. The policy gradient objective for a full sequence $y = (y_1,\dots,y_T)$ becomes:

$$
L^{PG}(\theta) = \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(\cdot|x)}\left[ \sum_{t=1}^T \log \pi_\theta(y_t|x,y_{<t}) \cdot \hat{A}_t \right],
$$

where $\hat{A}_t$ is estimated via Generalized Advantage Estimation (GAE) [source:spinningup:spinning-up-proximal-policy-optimization]. GAE computes $\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$ with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$; in the bandit limit $\gamma=1$, $\lambda=1$, and $r_t=0$ for $t<T$, this simplifies to $\hat{A}_t = r(x,y) - V(x)$ for all $t$. The value function $V_\phi(x)$ is trained with a squared-error loss $L^{VF} = (V_\phi(x) - r(x,y))^2$ [source:spinningup:spinning-up-proximal-policy-optimization].

**InstructGPT's PPO-ptx objective:** The InstructGPT paper [source:arxiv:2203.02155] formulates the combined PPO-ptx objective as:

$$
\text{objective} (\phi) = \mathbb{E}_{(x, y) \sim D_{\pi_{\phi}^{\mathrm{RL}}}} \left[ r_{\theta} (x, y) - \beta \log \left(\pi_{\phi}^{\mathrm{RL}} (y \mid x) / \pi^{\mathrm{SFT}} (y \mid x)\right) \right] + \gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} \left[ \log \left(\pi_{\phi}^{\mathrm{RL}} (x)\right) \right]
$$

where $\beta$ controls the KL penalty from the SFT model and $\gamma$ controls the strength of the pretraining gradients. This mixes the RL reward with a per-token KL penalty and a pretraining log-likelihood term to mitigate the alignment tax.

**Spinning Up implementation recipe:** The Spinning Up documentation [source:spinningup:spinning-up-proximal-policy-optimization] describes the PPO training loop as: (1) Collect trajectories $\mathcal{D}_k$ by running current policy $\pi_k$; (2) Compute rewards-to-go $\hat{R}_t$; (3) Compute advantage estimates $\hat{A}_t$ using GAE with current value function $V_{\phi_k}$; (4) Maximize PPO-Clip objective via stochastic gradient ascent (Adam); (5) Fit value function by minimizing MSE: $\phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T (V_{\phi}(s_t) - \hat{R}_t)^2$.

## Reward model integration

The reward model $r_\psi(x,y)$ is trained on pairwise human preferences. The InstructGPT paper [source:arxiv:2203.02155] trains the reward model on all $\binom{K}{2}$ comparisons from a single ranking task as a single batch element, with loss:

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

where $y_w$ is preferred over $y_l$. This increases training efficiency by using all pairwise comparisons from each ranking. The RM is trained to predict human preferences where labelers rank multiple model outputs for the same prompt from best to worst. The RM is typically initialized from the SFT model with a scalar head.

**RL framework mapping for LLMs:** Cameron Wolfe [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] maps the LLM process to RL components: **Policy** = the language model; **State** = the current textual input sequence; **Action** = the next token predicted by the model; **Reward** = the quality score provided by the reward model after a full sequence is produced.

## KL penalty: theory and practice

The KL penalty prevents the RL policy $\pi_\theta$ from drifting too far from the reference policy $\pi^{\text{ref}}$ (usually the SFT model), preserving language quality and preventing reward hacking. The modified reward is:

$$
r_{\text{total}}(x,y) = r_\psi(x,y) - \beta \, D_{\text{KL}}\bigl(\pi_\theta(\cdot|x) \,\|\, \pi^{\text{ref}}(\cdot|x)\bigr)
$$

**InstructGPT KL penalty:** The InstructGPT paper [source:arxiv:2203.02155] adds a per-token KL penalty from the SFT model to prevent over-optimization of the reward model, using the same formulation as the PPO-ptx objective above.

**Spinning Up early stopping:** The Spinning Up implementation [source:spinningup:spinning-up-proximal-policy-optimization] employs **early stopping** which halts gradient steps if the mean KL-divergence exceeds the `target_kl` threshold (typically 0.01 or 0.05), as clipping does not strictly guarantee the new policy remains close to the old one.

**KL divergence formulation:** Cameron Wolfe [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] emphasizes the use of **KL Divergence** to compare two probability distributions, $p$ and $q$, and to penalize reward hacking or excessively large policy updates. The KL divergence is formulated as the expected difference in log probabilities between two distributions:

$$
\text{KL}(p \parallel q) = \mathbb{E}_{x \sim p} [\log p(x) - \log q(x)]
$$

## PPO clipping mechanism

The clipped surrogate objective replaces the hard KL constraint of TRPO with a pessimistic bound on the policy ratio $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$ [source:spinningup:spinning-up-proximal-policy-optimization]:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\Bigl( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \Bigr) \right].
$$

For $\hat{A}_t > 0$, the objective increases with $r_t$ until $1+\epsilon$, then flattens; for $\hat{A}_t < 0$, it decreases until $1-\epsilon$, then flattens. This prevents destructive large updates when optimizing multiple epochs on the same batch. In LLMs, the ratio is computed per token:

$$
r_t(\theta) = \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t})},
$$

and the objective sums over tokens. The clipping threshold $\epsilon$ is typically 0.2.

**Spinning Up simplified clip objective:** The Spinning Up documentation [source:spinningup:spinning-up-proximal-policy-optimization] implements a simplified version:

$$
L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a), \; g(\epsilon, A^{\pi_{\theta_k}}(s,a)) \right)
$$

where:

$$
g(\epsilon, A) = \left\{ \begin{array}{ll} (1 + \epsilon) A & A \geq 0 \\ (1 - \epsilon) A & A < 0 \end{array} \right.
$$

If the advantage $A$ is positive, the objective is capped once $\pi_\theta(a|s) > (1+\epsilon)\pi_{\theta_k}(a|s)$. If the advantage is negative, the objective is capped once $\pi_\theta(a|s) < (1-\epsilon)\pi_{\theta_k}(a|s)$.

**TRPO to PPO evolution:** Cameron Wolfe [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] describes TRPO as improving upon VPG by constraining the KL divergence between old and updated policies, implemented via Taylor expansion approximation, Lagrangians/duality, and conjugate gradient algorithm to avoid matrix inversion. PPO is presented as a more robust, simpler, and more data-efficient alternative that avoids TRPO's complex analytical update rule.

**Clipping vs. penalty for LLMs:** All major LLM RLHF implementations use **both** clipping and a KL penalty simultaneously. The KL penalty acts on the divergence from the *reference* (SFT) model, while clipping constrains the step from the *old* (current iteration) policy. These serve different purposes: clipping ensures per-update stability; KL penalty anchors to the initial supervised model.

## PPO variants and practical modifications for LLMs

| Modification | Description | Sources |
|--------------|-------------|---------|
| **PPO-ptx** | Adds pretraining loss $\lambda_{\text{ptx}} \mathbb{E}[\log \pi_\theta(x)]$ to preserve pretrained capabilities | [source:arxiv:2203.02155] |
| **Reference model = SFT model** | Fixed $\pi^{\text{ref}}$ throughout training | [source:arxiv:2203.02155] |
| **All pairwise comparisons in RM batch** | Use all $\binom{K}{2}$ comparisons from each ranking as single batch element | [source:arxiv:2203.02155] |
| **Early stopping on KL** | Halt gradient steps if mean KL exceeds target_kl | [source:spinningup:spinning-up-proximal-policy-optimization] |

**Spinning Up typical hyperparameters [source:spinningup:spinning-up-proximal-policy-optimization]:**
- Clip Ratio ($\epsilon$): $0.1$ to $0.3$
- Target KL: $0.01$ or $0.05$ (used for early stopping)
- Discount Factor ($\gamma$): $0.99$
- GAE Lambda ($\lambda$): Close to $1$ (e.g., $0.97$)
- Learning Rates: $\pi_{lr} = 0.0003$ and $vf_{lr} = 0.001$

**Disagreement on entropy bonus:** The role of explicit entropy regularization in LLM PPO is not widely reported; implicit entropy control via KL penalty may suffice.

## Current status and trajectory

PPO remains the **default** RL optimizer for industrial RLHF pipelines (OpenAI, Anthropic, Google, Meta) as of 2024. The trajectory shows three phases:

1. **2019–2022: Establishment.** InstructGPT [source:arxiv:2203.02155] established PPO+KL as the canonical RLHF recipe. PPO was chosen for its stability over TRPO [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key]. InstructGPT demonstrated that the 1.3B parameter model was preferred over the 175B GPT-3 baseline, with the 175B InstructGPT preferred 85±3% over GPT-3 and 71±4% over few-shot GPT-3. Hallucination rate dropped from 41% to 21%, and toxicity reduced by ~25% when prompted to be respectful. InstructGPT significantly outperformed models fine-tuned on FLAN and T0 datasets: on the API prompt distribution, InstructGPT had a 73.4±2% winrate against the SFT baseline, while T0 and FLAN had 29.8±2% and 26.8±2%, respectively.

2. **2023: Engineering hardening.** Practice matured with documented techniques for stability (critic initialization, reward clipping, ptx loss, small buffers) addressing instability where policies overoptimize the RM.

3. **2023–present: Challenge from offline methods.** DPO and P3O demonstrate competitive or better KL-reward frontiers without online RL. RLP addresses RM staleness by retraining on policy samples. PPO's sample inefficiency (requires on-policy rollouts, multiple epochs) and infrastructure complexity (distributed rollout generation, critic training, GAE) motivate alternatives. However, PPO remains **necessary for**:
   - Iterative online RLHF where the RM is updated with fresh human data
   - Settings requiring per-token credit assignment (e.g., process supervision, tool use)
   - Cases where the reward is not purely preference-based (e.g., verifiable rewards, RL for reasoning)

**Hedge:** The field has not abandoned PPO; major labs still use it for their strongest models (InstructGPT, ChatGPT per [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key]). But the *proportion* of alignment research using PPO is declining as DPO, KTO, and RLAIF gain traction for static datasets. No source reports a large-scale ablation proving PPO's superiority over DPO when both are tuned optimally on the same data and compute budget. The "PPO vs. offline" debate is not settled by current benchmarks.

## Key takeaways

- PPO for LLMs adapts the clipped surrogate objective $L^{\text{CLIP}}$ to a contextual bandit: per-token ratios, sequence-level sparse reward, GAE reducing to $r(x,y)-V(x)$.
- Two distinct regularizers operate simultaneously: **clipping** ($\epsilon\approx0.2$) constrains the per-update policy ratio $r_t(\theta)$; **KL penalty** ($\beta\sim10^{-3}$ to $10^{-1}$) anchors to the frozen SFT model. They are not interchangeable.
- The reward model is trained with Bradley-Terry loss on pairwise preferences; InstructGPT uses all $\binom{K}{2}$ comparisons from each ranking as a single batch element [source:arxiv:2203.02155]. Its score is normalized and clipped before entering PPO. RM staleness under policy shift is a fundamental limitation addressed by RLP and iterative online RLHF.
- Practical stability requires pretraining gradient mixing (PPO-ptx) [source:arxiv:2203.02155].
- Token-level KL penalty (added to per-token reward) is mathematically equivalent to sequence-level KL for the total return but changes advantage estimation; most LLM implementations use token-level.
- PPO's sample inefficiency and infrastructure complexity motivate offline alternatives (DPO, P3O, RLP), but PPO remains the only method supporting fully online, iterative preference collection and per-token credit assignment.
- InstructGPT demonstrated that RLHF with PPO can make a 1.3B model outperform a 175B base model on human preference, reduce hallucination from 41% to 21%, and reduce toxicity by ~25% [source:arxiv:2203.02155].

## Related topics

- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md) — Offline alternative to PPO using the same preference data
- [GRPO (Group Relative Policy Optimization)](grpo.md) — Another PPO variant for LLMs
- [Reward modeling for LLMs](reward-modeling.md) — Details on RM architecture, training, and overoptimization
- [RL for reasoning models](rl-for-reasoning.md) — PPO applied to verifiable rewards (math, code)
- [Policy gradient methods for LLMs](policy-gradient-methods.md) — Broader policy gradient family
- [KL regularization in RLHF](kl-regularization.md) — Deep dive on KL penalty theory and adaptive control
- [MDP formulation of LLM generation](mdp-formulation.md) — Bandit vs. MDP perspectives
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md) — End-to-end system view
- [Reward model over-optimization](reward-model-overoptimization.md) — Pathology PPO exacerbates
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md) — Role of entropy bonus vs. KL
- [Distributed RL training for LLMs](distributed-rl-training.md) — Infrastructure for PPO rollouts and updates
- [Async and off-policy RL](async-and-off-policy-rl.md) — Alternatives to on-policy PPO
- [Rollout generation infrastructure](rollout-generation-infra.md) — Engineering of the environment interaction loop

## References
- [source:arxiv:2009.01325] [Simple, Scalable, and Effective Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2009.01325)
- [source:arxiv:2403.19279] [Fine-Tuning Language Models with Reward Learning on Policy](https://arxiv.org/abs/2403.19279)
- [source:arxiv:2310.00212] [Pairwise Proximal Policy Optimization (P3O)](https://arxiv.org/abs/2310.00212)
- [source:arxiv:2307.04964] [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964)
- [source:arxiv:1909.08593] [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- [source:arxiv:1706.03741] [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [source:arxiv:2204.05862] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [source:spinningup:spinning-up-proximal-policy-optimization] [Spinning Up: Proximal Policy Optimization (OpenAI)](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] [Proximal Policy Optimization (PPO): The Key to LLM Alignment (Cameron Wolfe)](https://cameronrwolfe.substack.com/p/proximal-policy-optimization-ppo)
