---
title: GRPO (Group Relative Policy Optimization)
maturity: comprehensive
updated: '2026-07-12'
sources:
- cameronrwolfe:group-relative-policy-optimization-grpo-
- arxiv:2402.03300
- arxiv:2405.02104
- snorkel:grpo-group-relative-policy-optimization-
- youtube:deepseek-s-grpo-group-relative-policy-op
- medium:the-math-behind-deepseek-a-deep-dive-int
- turingpost:what-is-grpo-group-relative-policy-optim
- arxiv:2511.03527
- aayushgarg:understanding-grpo-ppo-without-the-criti
open_questions:
- What is the theoretically optimal group size $G$ for GRPO, and how does it scale
  with task complexity and reward variance?
- Can GRPO's critic-free approach be extended to long-horizon agentic tasks (web navigation,
  tool use) without dense process rewards?
- How does the moving reference policy ($\pi_{\text{ref}} \leftarrow \pi_\theta$ each
  iteration) affect long-term training stability and mode collapse risk compared to
  a fixed SFT reference?
- Does the classical control result (critic-free methods fail without early termination)
  imply fundamental limits for GRPO on open-ended generation tasks without clear EOS?
---

Group Relative Policy Optimization (GRPO) is a critic-free policy gradient algorithm introduced by DeepSeek that replaces the value network in PPO with a group-relative baseline computed from multiple sampled responses per prompt. It has become the dominant reinforcement learning optimizer for reasoning-oriented large language models, particularly in the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm.

## Background and motivation

Proximal Policy Optimization (PPO) has been the workhorse of RLHF for LLMs, but its actor-critic architecture imposes substantial overhead: a separate critic model must be trained to estimate per-token state values, doubling memory requirements and introducing additional optimization instability [source:arxiv:2402.03300] [source:youtube:deepseek-s-grpo-group-relative-policy-op]. For reasoning tasks where a deterministic correctness signal is available at the end of a trajectory (e.g., math, code), the critic's per-step value estimates are both expensive and unnecessary [source:medium:the-math-behind-deepseek-a-deep-dive-int]. GRPO was designed to eliminate the critic entirely while retaining PPO's clipped surrogate objective and KL regularization, using the empirical distribution of rewards within a group of responses to the same prompt as a self-normalizing baseline [source:arxiv:2402.03300] [source:snorkel:grpo-group-relative-policy-optimization-].

**GRPO in the RLVR paradigm.** GRPO is primarily utilized within the framework of Reinforcement Learning with Verifiable Rewards (RLVR) to develop Large Reasoning Models (LRMs) that can dynamically "think" through problems before providing a final answer [source:turingpost:what-is-grpo-group-relative-policy-optim]. In RLVR, rewards are assigned via deterministic verifiers — ground-truth matching, code execution against test cases, or LLM judges verifying solution correctness — eliminating the need for a learned reward model and its associated reward hacking surface [source:turingpost:what-is-grpo-group-relative-policy-optim] [source:turingpost:what-is-grpo-group-relative-policy-optim]. This combination (GRPO + RLVR) has become the standard recipe for open reasoning models including DeepSeek-R1, Qwen-3, and Olmo-3, and is reported to underlie closed models such as OpenAI's o1-preview, o3, o4, and Gemini 3 [source:turingpost:what-is-grpo-group-relative-policy-optim].

**Critic-free RL in classical control.** A systematic study of GRPO in classical RL environments (CartPole, Acrobot, MountainCarContinuous, HalfCheetah, Humanoid) found that PPO with a learned value function substantially outperforms all critic-free baselines in long-horizon tasks [source:arxiv:2511.03527]. The only exception was CartPole, where critic-free methods often exceeded PPO, likely due to PPO overfitting or policy collapse in this short-horizon environment [source:arxiv:2511.03527]. This suggests the critic-free design may be particularly well-suited to the episodic, prompt-conditioned structure of LLM generation (where early termination / end-of-sequence provides a natural horizon) but less universally applicable to continuous control.

## Algorithm design

### Core idea

For each prompt $q$, the current policy $\pi_{\theta_{\text{old}}}$ generates a group of $G$ completions $\{o_i\}_{i=1}^G$. Each completion receives a scalar reward $r_i$ (from a reward model or verifiable ground truth). The group statistics — mean $\mu_r$ and standard deviation $\sigma_r$ — define a normalized advantage for every token in completion $i$ [source:arxiv:2402.03300]:

$$
\widetilde{r}_i = \frac{r_i - \mu_r}{\sigma_r}, \qquad \hat{A}_{i,t} = \widetilde{r}_i \quad \text{(outcome supervision)}
$$

This single scalar advantage is shared across all tokens $t$ of completion $o_i$, avoiding per-token value estimation entirely [source:youtube:deepseek-s-grpo-group-relative-policy-op] [source:snorkel:grpo-group-relative-policy-optimization-].

**Step-by-step process.** The GRPO update follows a concrete sequence [source:aayushgarg:understanding-grpo-ppo-without-the-criti]:
1. **Group Sampling:** For each prompt $q$, sample a group of $G$ completions $\{o_1, \dots, o_G\}$ from the old policy $\pi_{\theta_{\text{old}}}$.
2. **Reward Collection:** Assign each completion a reward $\{r_1, \dots, r_G\}$ via reward model or verifier.
3. **Advantage Computation:** Compute the group-relative advantage $\hat{A}_i = (r_i - \text{mean}(r)) / \text{std}(r)$ for each completion.
4. **Objective Optimization:** Update the policy using a clipped surrogate objective (similar to PPO) with the KL divergence penalty moved directly into the loss function.
5. **KL Estimation:** Use an unbiased, non-negative KL estimator to reduce variance and avoid log-probability underflow.

### Process supervision variant

When a process reward model (PRM) provides step-level rewards $r_i^{\text{index}(j)}$, GRPO normalizes them across the group and defines the advantage at token $t$ as the sum of normalized future step rewards [source:arxiv:2402.03300]:

$$
\widetilde{r}_i^{\text{index}(j)} = \frac{r_i^{\text{index}(j)} - \mu_R}{\sigma_R}, \qquad \hat{A}_{i,t} = \sum_{\text{index}(j) \ge t} \widetilde{r}_i^{\text{index}(j)}
$$

The DeepSeekMath paper reports that outcome supervision is preferred in practice because the marginal gains from process supervision "don't justify the overhead" [source:youtube:deepseek-s-grpo-group-relative-policy-op]. However, the aayushgarg source notes that outcome supervision (single final reward) may be insufficient or inefficient for supervising policies in complex mathematical tasks, necessitating process supervision to accelerate learning [source:aayushgarg:understanding-grpo-ppo-without-the-criti].

### Objective function

GRPO adopts PPO's clipped surrogate objective with the group-relative advantage and adds a direct KL penalty against a reference policy $\pi_{\text{ref}}$ [source:arxiv:2402.03300]:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min\left( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})} \hat{A}_{i,t}, \text{clip}\left( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}, 1-\varepsilon, 1+\varepsilon \right) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] \right\} \right]
$$

The KL term uses an unbiased estimator that avoids log-probability underflow [source:arxiv:2402.03300] [source:aayushgarg:understanding-grpo-ppo-without-the-criti]:

$$
\mathbb{D}_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - 1
$$

**Full objective in expectation form.** The aayushgarg source presents the objective as an expectation over prompts and groups [source:aayushgarg:understanding-grpo-ppo-without-the-criti]:

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D},\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(o|q)}\left[\frac{1}{G}\sum_{i=1}^G \left( \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}\hat{A}_i, \; \text{clip}(\cdot) \cdot \hat{A}_i\right) - \beta \, D_{\text{KL}}\left(\pi_\theta \| \pi_{\text{ref}}\right) \right)\right]
$$

### Hyperparameters from DeepSeekMath

| Parameter | Value |
|-----------|-------|
| Group size $G$ | 64 |
| Replay buffer | 10% historical data |

[source:arxiv:2402.03300]

**Classical control hyperparameters.** The arxiv:2511.03527 study used a learning rate of $2.5 \times 10^{-4}$, clipping coefficient $\epsilon = 0.2$, 4 epochs per iteration, and a 1M environment step budget [source:arxiv:2511.03527]. For GRPO they used Monte Carlo returns for full episodes ($N_{steps}=H$) and grouped trajectories from parallel environments [source:arxiv:2511.03527].

## Training procedure

### Iterative GRPO loop

DeepSeekMath employs an iterative RL pipeline [source:arxiv:2402.03300]:
1. **Initialize**: Policy $\pi_\theta$ = SFT model; reference $\pi_{\text{ref}}$ = SFT model; reward model $r_\varphi$ trained on base model.
2. **Exploration**: For each prompt in the training set (~144K GSM8K/MATH questions), sample $G=64$ completions from $\pi_{\theta_{\text{old}}}$.
3. **Reward**: Score completions with $r_\varphi$ (or verifier).
4. **Update**: Compute GRPO objective and update $\pi_\theta$ for one step.
5. **Iterate**: Refresh reward model training data using new policy samples; retrain $r_\varphi$ with 10% replay; set $\pi_{\text{ref}} \leftarrow \pi_\theta$; repeat.

The reference model is updated to the current policy at each iteration, unlike standard PPO where $\pi_{\text{ref}}$ is typically frozen as the initial SFT model [source:arxiv:2402.03300]. This moving reference is a notable design choice that effectively implements a trust region around the most recent policy rather than the original SFT model.

### RLVR instantiation

In the RLVR setting (e.g., DeepSeek-R1), the reward model is replaced by a deterministic verifier (unit tests, exact match, symbolic equivalence) [source:cameronrwolfe:group-relative-policy-optimization-grpo-] [source:snorkel:grpo-group-relative-policy-optimization-]. The pipeline becomes:
1. Curate verifiable dataset (math, code) with ground-truth answers.
2. Enforce parsable output format (e.g., `\\boxed{}` for math).
3. Run iterative GRPO with verifier rewards.
4. No reward model training needed; eliminates reward hacking surface [source:turingpost:what-is-grpo-group-relative-policy-optim].

**RLVR reward computation details.** Verifiable rewards in RLVR are obtained via [source:turingpost:what-is-grpo-group-relative-policy-optim]:
- **Ground Truth Matching:** String matching between predicted answer and known correct answer.
- **Execution Feedback:** Sandbox code execution assessed via test cases.
- **LLM Judges:** A separate LLM verifies if a solution attempt matches ground truth, more robust than strict parsing engines.

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

**Memory efficiency.** GRPO reduces memory consumption by eliminating the critic model, which in standard PPO would be one of four resident models (policy, critic, reference, reward model) [source:aayushgarg:understanding-grpo-ppo-without-the-criti].

**Disagreement**: The YouTube source states GRPO uses "typically 4 to 8" group samples [source:youtube:deepseek-s-grpo-group-relative-policy-op], while the DeepSeekMath paper uses $G=64$ [source:arxiv:2402.03300] and Snorkel reports "typically 8 to 64" [source:snorkel:grpo-group-relative-policy-optimization-]. The paper's 64 is specific to their math RL setup; smaller groups may suffice for verifiable rewards with lower variance. Cameron Wolfe's blog implies GRPO is now "prevalent" for RLVR [source:cameronrwolfe:group-relative-policy-optimization-grpo-], but the YouTube source cautions that GRPO's simplifications "might introduce more variance" compared to full actor-critic [source:youtube:deepseek-s-grpo-group-relative-policy-op]. The trade-off between variance reduction (critic) and compute/memory savings (no critic) is not definitively settled in the sources.

**NEW DISAGREEMENT — Classical control vs. LLM results.** The arxiv:2511.03527 study finds that in classical RL benchmarks, PPO with a learned critic **substantially outperforms** GRPO and other critic-free baselines on long-horizon tasks (Acrobot, MountainCarContinuous, HalfCheetah, Humanoid) [source:arxiv:2511.03527]. LLM generation has a natural horizon (EOS token), which may explain why GRPO works well there despite failing in continuing control tasks. The study identifies **early termination** as a critical factor: environments with natural termination (CartPole, Acrobot) allow critic-free methods to extract learning signals from the return distribution, while continuing tasks without early termination (HalfCheetah) dilute the signal [source:arxiv:2511.03527].

**Discount factor sensitivity.** The classical control study found $\gamma=0.99$ optimal for most environments, but **HalfCheetah** (the only environment without early termination) performed worst at $\gamma=1$ and optimal at $\gamma=0.9$ [source:arxiv:2511.03527]. This suggests the discount factor interacts critically with horizon structure in critic-free methods.

**Group size effects.** Contrary to expectations, smaller group sizes ($G=8$) generally outperformed larger ones ($G=128$) in classical control, even when controlling for update frequency [source:arxiv:2511.03527]. This may reflect the grouping strategy (mixing unrelated episodes from parallel environments) introducing noise that larger groups amplify.

## Variants and extensions

### GiGPO (Group-in-Group Policy Optimization)

Snorkel reports a successor, GiGPO, which "reports gains of more than 12% over GRPO on ALFWorld and more than 9% on WebShop at the same memory cost" [source:snorkel:grpo-group-relative-policy-optimization-]. The mechanism is not detailed in the provided sources.

### Comparison to RLOO (REINFORCE Leave-One-Out)

Unlike REINFORCE Leave-One-Out (RLOO), which uses a baseline excluding the current sample and lacks clipping, GRPO incorporates PPO-style clipping and reward normalization [source:aayushgarg:understanding-grpo-ppo-without-the-criti].

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

DeepSeek-R1 (not detailed in sources but cited as GRPO-scaled) achieves strong reasoning benchmarks [source:cameronrwolfe:group-relative-policy-optimization-grpo-].

**Classical control benchmarks (1M steps).** PPO with learned critic vs. GRPO (group size sweep) [source:arxiv:2511.03527]:
- **CartPole-v1**: Critic-free methods (including GRPO) often exceed PPO (PPO may overfit/collapse).
- **Acrobot-v1**: PPO substantially outperforms all critic-free baselines.
- **MountainCarContinuous-v0**: PPO substantially outperforms all critic-free baselines.
- **HalfCheetah-v4**: PPO substantially outperforms; GRPO performs worst at $\gamma=1$, best at $\gamma=0.9$; smaller groups ($G=8$) better than larger.
- **Humanoid-v4**: PPO substantially outperforms all critic-free baselines.

The study concludes that early termination creates a natural separation between successful and unsuccessful trajectories, allowing critic-free methods to extract learning signals that are otherwise diluted in continuing tasks [source:arxiv:2511.03527].

## Limitations and open challenges

1. **Long-horizon credit assignment**: GRPO assigns a single terminal reward (or summed process rewards) to all tokens in a completion. For multi-step agentic tasks (e.g., web navigation, tool use), "every step receives the same negative signal, making it difficult for the model to identify which specific actions led to failure" [source:snorkel:grpo-group-relative-policy-optimization-]. This causes noisy gradients and slow/failed convergence.

2. **Single-response verifiable outcome bias**: GRPO "works best when a task involves a single response with one verifiable outcome" [source:snorkel:grpo-group-relative-policy-optimization-]. Multi-turn, open-ended, or subjective tasks are ill-suited.

3. **Group size vs. variance trade-off**: Small groups yield high-variance advantage estimates; large groups increase compute linearly. The optimal $G$ is not theoretically characterized in the sources. **New finding:** In classical control, smaller groups ($G=8$) outperformed larger ones ($G=128$), possibly due to grouping strategy mixing unrelated episodes [source:arxiv:2511.03527].

4. **Moving reference model**: Updating $\pi_{\text{ref}} \leftarrow \pi_\theta$ each iteration [source:arxiv:2402.03300] deviates from standard PPO's fixed reference. The long-term effects on mode collapse or catastrophic forgetting are not analyzed in the sources.

5. **Geometry/theorem-proving weakness**: DeepSeekMath remains weaker on geometry and formal theorem proving, "potentially due to data selection bias" [source:arxiv:2402.03300]. GRPO does not remedy data gaps.

6. **Domain restriction (RLVR)**: RLVR (and by extension, the most common use of GRPO) requires "verifiable domains" such as mathematics or coding where a ground truth or deterministic verification rule exists [source:turingpost:what-is-grpo-group-relative-policy-optim]. Expansion to non-verifiable domains remains an open challenge.

7. **Verification robustness**: Simple string matching is often insufficient for evaluating correctness, necessitating more complex validation logic or the use of LLM judges to capture format variations [source:turingpost:what-is-grpo-group-relative-policy-optim].

8. **Grouping strategy limitations**: In classical control, groups are formed from trajectories collected concurrently from parallel environments, mixing potentially unrelated episodes — unlike prompt-based grouping in LLMs where all completions share the same prompt [source:arxiv:2511.03527]. This may be suboptimal and contribute to the poor performance of GRPO in continuing control tasks.

9. **Hyperparameter exploration gaps**: The classical control study only partially explored the hyperparameter space regarding mini-batches, entropy regularization, and learning rates [source:arxiv:2511.03527]. Results may differ in sparse reward or partially observable domains.

## Current status and trajectory

GRPO is **rising rapidly and becoming the default** for reasoning-model RL, especially in the RLVR paradigm. It is the core optimizer behind DeepSeek-R1 and has been adopted by open-source replication efforts [source:cameronrwolfe:group-relative-policy-optimization-grpo-]. The critic-free design aligns perfectly with verifiable rewards, eliminating both the reward model and the value network — a "refreshingly simple—and effective" combination [source:cameronrwolfe:group-relative-policy-optimization-grpo-].

However, the field is **not settled on GRPO as a universal replacement for PPO**. For preference-based RLHF (non-verifiable rewards), PPO with a critic remains common because the reward model's noise and the lack of ground-truth correctness make group-relative normalization less reliable. Cameron Wolfe notes "ongoing research into tweaking GRPO, scaling RLVR training, expanding to non-verifiable domains via rubrics" [source:cameronrwolfe:group-relative-policy-optimization-grpo-], indicating active evolution rather than convergence. GiGPO's reported gains on agentic benchmarks [source:snorkel:grpo-group-relative-policy-optimization-] suggest the basic GRPO formulation may be insufficient for long-horizon tasks. The technique is **dominant in its niche (verifiable-reward reasoning) but not yet proven as a general-purpose RLHF optimizer**.

**Critical caveat from classical control:** The arxiv:2511.03527 study demonstrates that GRPO's critic-free design **fails to match PPO in long-horizon continuous control tasks without early termination** [source:arxiv:2511.03527]. Since LLM generation has a natural horizon (EOS token), this limitation may not transfer directly, but it warns against assuming GRPO is universally superior to actor-critic methods.

## Key takeaways

- GRPO removes the critic network from PPO by using the mean and standard deviation of rewards across a group of $G$ completions per prompt as a self-normalizing baseline.
- The advantage for completion $i$ is $\hat{A}_i = (r_i - \mu_r)/\sigma_r$ (outcome supervision) or a sum of normalized future step rewards (process supervision).
- The objective combines PPO's clipped surrogate with a direct KL penalty using an unbiased estimator; the reference policy is updated to the current policy each iteration in DeepSeek's implementation.
- Group size $G=64$ was used in DeepSeekMath; smaller groups (8–16) are common in RLVR replications. The variance/compute trade-off is not theoretically resolved. **New finding:** In classical control, $G=8$ outperformed larger groups, possibly due to grouping strategy artifacts [source:arxiv:2511.03527].
- GRPO is the de facto standard for **RLVR** (math, code with verifiable rewards) but less established for **preference-based RLHF** where reward noise and subjectivity favor a learned critic.
- Main limitations: poor credit assignment for long-horizon tasks, sensitivity to reward noise, restriction to verifiable or reliably modeled rewards, and (per classical control study) dependence on early termination / episodic structure for effective learning.
- Iterative GRPO with reward model replay (10%) and moving reference is the full DeepSeekMath recipe; RLVR drops the reward model entirely.
- **Models using GRPO:** DeepSeek-R1, Qwen-3, Olmo-3 (open); OpenAI o1-preview, o3, o4, Gemini 3 (reported closed) [source:turingpost:what-is-grpo-group-relative-policy-optim].

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
- [source:cameronrwolfe:group-relative-policy-optimization-grpo-] [Group Relative Policy Optimization (GRPO) - Deep (Learning) Focus](https://cameronrwolfe.substack.com/p/grpo)
- [source:arxiv:2402.03300] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (RLVR/GRPO)](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2405.02104] [DeepSeek-RLHF: A Strong and Customizable LLM with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2405.02104)
- [source:snorkel:grpo-group-relative-policy-optimization-] [GRPO (Group Relative Policy Optimization), explained - Snorkel AI](https://snorkel.ai/grpo/)
- [source:youtube:deepseek-s-grpo-group-relative-policy-op] [DeepSeek's GRPO (Group Relative Policy Optimization) - YouTube](https://www.youtube.com/watch?v=xT4jxQUl0X8)
- [source:medium:the-math-behind-deepseek-a-deep-dive-int] [The Math Behind DeepSeek: A Deep Dive into Group Relative ...](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba)
- [source:turingpost:what-is-grpo-group-relative-policy-optim] [What Is GRPO? Group Relative Policy Optimization Explained](https://www.turingpost.com/p/grpo)
- [source:arxiv:2511.03527] [Learning Without Critics? Revisiting GRPO in Classical Control and LLMs](https://arxiv.org/abs/2511.03527)
- [source:aayushgarg:understanding-grpo-ppo-without-the-criti] [Understanding GRPO: PPO without the Critic](https://aayushgarg.dev/posts/2026-01-01-understanding-grpo.html)
