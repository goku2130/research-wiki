---
title: GRPO (Group Relative Policy Optimization)
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2402.03300
- cameronrwolfe:group-relative-policy-optimization-grpo-
- medium:the-math-behind-deepseek-a-deep-dive-int
- snorkel:grpo-group-relative-policy-optimization-
- youtube:deepseek-s-grpo-group-relative-policy-op
- arxiv:2402.03300
- arxiv:2405.02104
open_questions:
- What is the theoretically optimal group size $G$ as a function of reward variance,
  sequence length, and policy entropy? No source provides a principled derivation.
- Does the moving reference policy ($\pi_{\text{ref}} \leftarrow \pi_\theta$ each
  iteration) prevent or accelerate mode collapse compared to a fixed SFT reference?
  The sources do not analyze this.
- Can GRPO's group-relative advantage be made robust to noisy/uncalibrated rewards
  (e.g., via trimmed means, median-of-means, or learned uncertainty weighting)? The
  DeepSeekMath paper flags reward noise as a critical gap but offers no solution.
- How should GRPO be modified for long-horizon agentic tasks where credit assignment
  over thousands of tokens is required? GiGPO claims gains but its mechanism is not
  described in the sources.
---

Group Relative Policy Optimization (GRPO) is a critic-free policy gradient algorithm introduced by DeepSeek that replaces the value network in PPO with a group-relative baseline computed from multiple sampled responses per prompt. It has become the dominant reinforcement learning optimizer for reasoning-oriented large language models, particularly in the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm.

## Background and motivation

Proximal Policy Optimization (PPO) has been the workhorse of RLHF for LLMs, but its actor-critic architecture imposes substantial overhead: a separate critic model must be trained to estimate per-token state values, doubling memory requirements and introducing additional optimization instability [source:arxiv:2402.03300] [source:youtube:deepseek-s-grpo-group-relative-policy-op]. For reasoning tasks where a deterministic correctness signal is available at the end of a trajectory (e.g., math, code), the critic's per-step value estimates are both expensive and unnecessary [source:medium:the-math-behind-deepseek-a-deep-dive-int]. GRPO was designed to eliminate the critic entirely while retaining PPO's clipped surrogate objective and KL regularization, using the empirical distribution of rewards within a group of responses to the same prompt as a self-normalizing baseline [source:arxiv:2402.03300] [source:snorkel:grpo-group-relative-policy-optimization-].

## Algorithm design

### Core idea

For each prompt $q$, the current policy $\pi_{\theta_{\text{old}}}$ generates a group of $G$ completions $\{o_i\}_{i=1}^G$. Each completion receives a scalar reward $r_i$ (from a reward model or verifiable ground truth). The group statistics — mean $\mu_r$ and standard deviation $\sigma_r$ — define a normalized advantage for every token in completion $i$ [source:arxiv:2402.03300]:

$$
\widetilde{r}_i = \frac{r_i - \mu_r}{\sigma_r}, \qquad \hat{A}_{i,t} = \widetilde{r}_i \quad \text{(outcome supervision)}
$$

This single scalar advantage is shared across all tokens $t$ of completion $o_i$, avoiding per-token value estimation entirely [source:youtube:deepseek-s-grpo-group-relative-policy-op] [source:snorkel:grpo-group-relative-policy-optimization-].

### Process supervision variant

When a process reward model (PRM) provides step-level rewards $r_i^{\text{index}(j)}$, GRPO normalizes them across the group and defines the advantage at token $t$ as the sum of normalized future step rewards [source:arxiv:2402.03300]:

$$
\widetilde{r}_i^{\text{index}(j)} = \frac{r_i^{\text{index}(j)} - \mu_R}{\sigma_R}, \qquad \hat{A}_{i,t} = \sum_{\text{index}(j) \ge t} \widetilde{r}_i^{\text{index}(j)}
$$

The DeepSeekMath paper reports that outcome supervision is preferred in practice because the marginal gains from process supervision "don't justify the overhead" [source:youtube:deepseek-s-grpo-group-relative-policy-op].

### Objective function

GRPO adopts PPO's clipped surrogate objective with the group-relative advantage and adds a direct KL penalty against a reference policy $\pi_{\text{ref}}$ [source:arxiv:2402.03300]:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right\} \right]
$$

The KL term uses an unbiased estimator that avoids log-probability underflow [source:arxiv:2402.03300]:

$$
\mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - 1
$$

### Hyperparameters from DeepSeekMath

| Parameter | Value |
|-----------|-------|
| Group size $G$ | 64 |
| KL coefficient $\beta$ | 0.04 |
| Policy learning rate | 1e-6 |
| PPO clip $\varepsilon$ | (not explicitly stated; typical 0.2) |
| Max sequence length | 1024 |
| Training batch size | 1024 |
| Reward model LR | 2e-5 |
| Replay buffer | 10% historical data |

[source:arxiv:2402.03300]

## Training procedure

### Iterative GRPO loop

DeepSeekMath employs an iterative RL pipeline [source:arxiv:2402.03300]:
1. **Initialize**: Policy $\pi_\theta$ = SFT model; reference $\pi_{\text{ref}}$ = SFT model; reward model $r_\varphi$ trained on base model.
2. **Exploration**: For each prompt in the training set (~144K GSM8K/MATH questions), sample $G=64$ completions from $\pi_{\theta_{\text{old}}}$.
3. **Reward**: Score completions with $r_\varphi$ (or verifier).
4. **Update**: Compute GRPO objective and update $\pi_\theta$ for one step.
5. **Iterate**: Refresh reward model training data using new policy samples; retrain $r_\varphi$ with 10% replay; set $\pi_{\text{ref}} \leftarrow \pi_\theta$; repeat.

The reference model is updated to the current policy at each iteration, unlike standard PPO where $\pi_{\text{ref}}$ is typically frozen as the initial SFT model [source:arxiv:2402:03300]. This moving reference is a notable design choice that effectively implements a trust region around the most recent policy rather than the original SFT model.

### RLVR instantiation

In the RLVR setting (e.g., DeepSeek-R1), the reward model is replaced by a deterministic verifier (unit tests, exact match, symbolic equivalence) [source:cameronrwolfe:group-relative-policy-optimization-grpo-] [source:snorkel:grpo-group-relative-policy-optimization-]. The pipeline becomes:
1. Curate verifiable dataset (math, code) with ground-truth answers.
2. Enforce parsable output format (e.g., `\\boxed{}` for math).
3. Run iterative GRPO with verifier rewards.
4. No reward model training needed; eliminates reward hacking surface [source:cameronrwolfe:group-relative-policy-optimization-grpo-].

## Comparison to PPO

| Aspect | PPO (standard RLHF) | GRPO |
|--------|---------------------|------|
| Critic/value network | Required (per-token $V(s_t)$) | **None** — group statistics as baseline |
| Advantage estimation | GAE($\lambda$) over trajectory | Single scalar per completion (outcome) or sum of future step rewards (process) |
| Baseline | Learned $V_\phi(s_t)$ | $\mu_r$ of group |
| Normalization | Advantage normalization optional | Built-in: $(r_i - \mu_r)/\sigma_r$ |
| KL penalty | Often via critic or separate term | Direct KL term in objective with unbiased estimator |
| Memory | Policy + critic + ref + reward model | Policy + ref + reward model (or verifier) |
| Compute per step | Higher (critic forward/backward) | Lower (no critic) |
| Reward source | Learned RM (preference pairs) | Learned RM **or** verifiable ground truth |

[source:arxiv:2402.03300] [source:youtube:deepseek-s-grpo-group-relative-policy-op] [source:medium:the-math-behind-deepseek-a-deep-dive-int] [source:snorkel:grpo-group-relative-policy-optimization-]

**Disagreement**: The YouTube source states GRPO uses "typically 4 to 8" group samples [source:youtube:deepseek-s-grpo-group-relative-policy-op], while the DeepSeekMath paper uses $G=64$ [source:arxiv:2402.03300] and Snorkel reports "typically 8 to 64" [source:snorkel:grpo-group-relative-policy-optimization-]. The paper's 64 is specific to their math RL setup; smaller groups may suffice for verifiable rewards with lower variance. Cameron Wolfe's blog implies GRPO is now "prevalent" for RLVR [source:cameronrwolfe:group-relative-policy-optimization-grpo-], but the YouTube source cautions that GRPO's simplifications "might introduce more variance" compared to full actor-critic [source:youtube:deepseek-s-grpo-group-relative-policy-op]. The trade-off between variance reduction (critic) and compute/memory savings (no critic) is not definitively settled in the sources.

## Variants and extensions

### GiGPO (Group-in-Group Policy Optimization)

Snorkel reports a successor, GiGPO, which "reports gains of more than 12% over GRPO on ALFWorld and more than 9% on WebShop at the same memory cost" [source:snorkel:grpo-group-relative-policy-optimization-]. The mechanism is not detailed in the provided sources.

### Integration with other techniques

- **Curriculum learning**: Cameron Wolfe notes ongoing research into "curriculum learning" for RLVR/GRPO [source:cameronrwolfe:group-relative-policy-optimization-grpo-].
- **Hybrid rewards**: Combining verifiable and non-verifiable (preference) rewards is an active direction [source:cameronrwolfe:group-relative-policy-optimization-grpo-].
- **Rubrics for non-verifiable domains**: Expanding beyond math/code via structured rubrics evaluated by LLM judges [source:cameronrwolfe:group-relative-policy-optimization-grpo-].

## Empirical results

DeepSeekMath 7B (GRPO on top of SFT) [source:arxiv:2402.03300]:

| Benchmark | SFT (Instruct) | +GRPO (RL) | Δ |
|-----------|----------------|------------|---|
| MATH (CoT, Top1) | 46.8% | **51.7%** | +4.9% |
| MATH (Self-consistency@64) | — | **60.9%** | — |
| GSM8K (CoT) | 82.9% | **88.2%** | +5.3% |
| CMATH (CoT) | 84.6% | **88.8%** | +4.2% |
| MATH (Tool-integrated) | 57.4% | **58.8%** | +1.4% |

The paper notes RL primarily improves **Maj@K** (majority voting) rather than **Pass@K** (single-sample accuracy), suggesting GRPO "makes the output distribution more robust by boosting correct responses from TopK rather than fundamentally enhancing core capabilities" [source:arxiv:2402.03300].

DeepSeek-R1 (not detailed in sources but cited as GRPO-scaled) achieves strong reasoning benchmarks [source:cameronrwolfe:group-relative-policy-optimization-grpo-].

## Limitations and open challenges

1. **Long-horizon credit assignment**: GRPO assigns a single terminal reward (or summed process rewards) to all tokens in a completion. For multi-step agentic tasks (e.g., web navigation, tool use), "every step receives the same negative signal, making it difficult for the model to identify which specific actions led to failure" [source:snorkel:grpo-group-relative-policy-optimization-]. This causes noisy gradients and slow/failed convergence.

2. **Single-response verifiable outcome bias**: GRPO "works best when a task involves a single response with one verifiable outcome" [source:snorkel:grpo-group-relative-policy-optimization-]. Multi-turn, open-ended, or subjective tasks are ill-suited.

3. **Reward noise sensitivity**: The DeepSeekMath paper warns that "all current methods fully trust the reward function signal, which is problematic as reward signals can be unreliable (e.g., PRM800K dataset has ~20% incorrect annotations)" [source:arxiv:2402.03300]. GRPO has no built-in robustness to label noise.

4. **Group size vs. variance trade-off**: Small groups yield high-variance advantage estimates; large groups increase compute linearly. The optimal $G$ is not theoretically characterized in the sources.

5. **Moving reference model**: Updating $\pi_{\text{ref}} \leftarrow \pi_\theta$ each iteration [source:arxiv:2402.03300] deviates from standard PPO's fixed reference. The long-term effects on mode collapse or catastrophic forgetting are not analyzed in the sources.

6. **Geometry/theorem-proving weakness**: DeepSeekMath remains weaker on geometry and formal theorem proving, "potentially due to data selection bias" [source:arxiv:2402.03300]. GRPO does not remedy data gaps.

## Current status and trajectory

GRPO is **rising rapidly and becoming the default** for reasoning-model RL, especially in the RLVR paradigm. It is the core optimizer behind DeepSeek-R1 and has been adopted by open-source replication efforts (e.g., Open-R1, various Hugging Face community trains) [source:cameronrwolfe:group-relative-policy-optimization-grpo-]. The critic-free design aligns perfectly with verifiable rewards, eliminating both the reward model and the value network — a "refreshingly simple—and effective" combination [source:cameronrwolfe:group-relative-policy-optimization-grpo-].

However, the field is **not settled on GRPO as a universal replacement for PPO**. For preference-based RLHF (non-verifiable rewards), PPO with a critic remains common because the reward model's noise and the lack of ground-truth correctness make group-relative normalization less reliable [source:youtube:deepseek-s-grpo-group-relative-policy-op] [source:medium:the-math-behind-deepseek-a-deep-dive-int]. Cameron Wolfe notes "ongoing research into tweaking GRPO, scaling RLVR training, expanding to non-verifiable domains via rubrics" [source:cameronrwolfe:group-relative-policy-optimization-grpo-], indicating active evolution rather than convergence. GiGPO's reported gains on agentic benchmarks [source:snorkel:grpo-group-relative-policy-optimization-] suggest the basic GRPO formulation may be insufficient for long-horizon tasks. The technique is **dominant in its niche (verifiable-reward reasoning) but not yet proven as a general-purpose RLHF optimizer**.

## Key takeaways

- GRPO removes the critic network from PPO by using the mean and standard deviation of rewards across a group of $G$ completions per prompt as a self-normalizing baseline.
- The advantage for completion $i$ is $\hat{A}_i = (r_i - \mu_r)/\sigma_r$ (outcome supervision) or a sum of normalized future step rewards (process supervision).
- The objective combines PPO's clipped surrogate with a direct KL penalty using an unbiased estimator; the reference policy is updated to the current policy each iteration in DeepSeek's implementation.
- Group size $G=64$ was used in DeepSeekMath; smaller groups (8–16) are common in RLVR replications. The variance/compute trade-off is not theoretically resolved.
- GRPO is the de facto standard for **RLVR** (math, code with verifiable rewards) but less established for **preference-based RLHF** where reward noise and subjectivity favor a learned critic.
- Main limitations: poor credit assignment for long-horizon tasks, sensitivity to reward noise, and restriction to verifiable or reliably modeled rewards.
- Iterative GRPO with reward model replay (10%) and moving reference is the full DeepSeekMath recipe; RLVR drops the reward model entirely.

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md) — the actor-critic baseline GRPO replaces
- [RL for reasoning models](rl-for-reasoning.md) — the primary application domain for GRPO
- [Verifiable rewards (RLVR)](verifiable-rewards.md) — the reward paradigm that makes critic-free RL viable
- [Reward modeling for LLMs](reward-modeling.md) — still used in non-RLVR GRPO variants
- [KL regularization in RLHF](kl-regularization.md) — GRPO's direct KL penalty vs. PPO's approaches
- [RLHF/PPO pipeline](rlhf-ppo-pipeline.md) — end-to-end comparison of pipelines
- [Distributed RL training for LLMs](distributed-rl-training.md) — GRPO's memory savings impact distributed strategies
- [Rollout generation infrastructure](rollout-generation-infra.md) — group sampling ($G \times$ batch) requirements
- [RL for math and code](rl-for-math-and-code.md) — canonical GRPO use cases
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md) — KL regularization and moving reference effects
- [Process vs outcome reward models](process-vs-outcome-rewards.md) — GRPO's two supervision modes
- [Reward hacking in RLHF](reward-hacking.md) — RLVR+GRPO mitigates this by design
- [Self-improvement and self-play RL](self-improvement-and-self-play.md) — iterative GRPO as self-improvement loop

## References
- [source:arxiv:2402.03300] [[PDF] arXiv:2402.03300v3 [cs.CL] 27 Apr 2024](https://arxiv.org/pdf/2402.03300)
- [source:cameronrwolfe:group-relative-policy-optimization-grpo-] [Group Relative Policy Optimization (GRPO) - Deep (Learning) Focus](https://cameronrwolfe.substack.com/p/grpo)
- [source:medium:the-math-behind-deepseek-a-deep-dive-int] [The Math Behind DeepSeek: A Deep Dive into Group Relative ...](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba)
- [source:snorkel:grpo-group-relative-policy-optimization-] [GRPO (Group Relative Policy Optimization), explained - Snorkel AI](https://snorkel.ai/grpo/)
- [source:youtube:deepseek-s-grpo-group-relative-policy-op] [DeepSeek's GRPO (Group Relative Policy Optimization) - YouTube](https://www.youtube.com/watch?v=xT4jxQUl0X8)
- [source:arxiv:2402.03300] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2405.02104] [DeepSeek-RLHF: A Strong and Customizable LLM with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2405.02104)
