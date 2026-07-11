---
id: openrlhf:openrlhf-kl-penalty-documentation
type: web
title: 'OpenRLHF: KL Penalty Documentation'
url: https://openrlhf.readthedocs.io/en/latest/agent_training.html
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

# OpenRLHF RL Training Framework

OpenRLHF provides a unified agent execution pipeline for Reinforcement Learning from Human Feedback (RLHF), designed to decouple the experience collection (execution mode), the policy update mechanism (RL algorithm), and the hardware orchestration (pipeline).

### Core Problem
The framework addresses the need for a flexible, scalable RLHF system that can handle diverse task structures—ranging from simple single-turn prompt-response pairs to complex multi-turn agent interactions—while maintaining compatibility with various advantage estimators and high-throughput inference engines like vLLM.

### Method and Recipe
The RL training process is configured across three orthogonal axes:

1.  **Execution Mode Selection**:
    *   **Single-Turn**: A prompt produces one response and one reward. Rewards are sourced from a trained reward model, a remote HTTP server, or a local Python reward function (Reinforced Fine-Tuning).
    *   **Multi-Turn**: Implemented via `AgentInstanceBase` and `MultiTurnAgentExecutor`. The process follows a cycle: `reset` the environment $\rightarrow$ model generates action $\rightarrow$ environment returns feedback, reward, and a `done` flag $\rightarrow$ repeat until termination.

2.  **Algorithm Configuration**:
    The policy is updated using an advantage estimator selected via `--algo.advantage.estimator`:
    *   **PPO**: Uses Generalized Advantage Estimation (GAE) with a full critic and a clipped surrogate objective.
    *   **REINFORCE++**: A critic-free approach utilizing PPO clipping, KL penalties, and reward normalization.
    *   **REINFORCE++-baseline**: Uses mean reward as a baseline; specifically recommended for RLVR and reasoning tasks.
    *   **RLOO**: Employs a leave-one-out baseline with PPO clipping and per-token KL.
    *   **GRPO / Dr. GRPO**: Uses per-group mean/std normalization (Dr. GRPO omits the local standard deviation normalization).

3.  **Optimization and Tuning**:
    *   **Optimizer**: Supports Adam or Muon per entity (actor/critic). Muon is used for 2-D weight groups, while Adam handles 1-D parameters and embeddings.
    *   **KL Control**: KL divergence is applied either as a reward penalty (using estimator $k1$) or as a separate loss term (using estimators $k2$ or $k3$ for GRPO).
    *   **Off-Policy Correction**: To resolve log-prob discrepancies between vLLM rollouts and the trainer, importance-sampling (IS) correction is applied via TIS (clamping), ICEPOP (filtering), or Seq-Mask-TIS.
    *   **Dynamic Sampling (DAPO)**: Multiple completions are generated per prompt, and a subset is retained based on reward scores.

### Key Formulas
The **ICEPOP** off-policy correction is implemented as a hard mask:

$$
\text{vllm\_is} = \exp(\text{old\_log\_probs} - \text{rollout\_log\_probs})
$$

$$
\text{mask} = (\text{vllm\_is} \ge \text{low}) \land (\text{vllm\_is} \le \text{high})
$$

$$
\text{vllm\_is} = \text{vllm\_is} \times \text{mask}
$$

### Quantitative Results and Defaults
*   **PPO Defaults**: Clip range $\epsilon = 0.2$; Critic value clip $= 0.5$.
*   **KL Defaults**: Initial coefficient $= 0.01$.
*   **Reward Shaping**: Default raw reward clamp range is $[-10, 10]$.
*   **IS Correction**: Default thresholds are $[0.5, 5.0]$.
*   **Example Configuration (Qwen3-4B-Thinking)**:
    *   **Algorithm**: REINFORCE++-baseline.
    *   **Hyperparameters**: Actor learning rate $5\text{e-}7$; KL initial coefficient $1\text{e-}5$.
    *   **Batching**: Rollout batch size $128$; Training batch size $1024$; Samples per prompt $n=8$.
    *   **Sequence Length**: Max new tokens $64,000$; Max length $74,240$.

### Limitations
*   **LoRA**: The Ray + vLLM integration does not currently support LoRA.
*   **Muon Optimizer**: Incompatible with `--ds.adam_offload` and requires DeepSpeed $\ge 0.18.2$.
