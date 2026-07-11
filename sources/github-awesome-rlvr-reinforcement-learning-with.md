---
id: github:awesome-rlvr-reinforcement-learning-with
type: web
title: Awesome RLVR — Reinforcement Learning with Verifiable Rewards
url: https://github.com/opendilab/awesome-RLVR
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

# Reinforcement Learning with Verifiable Rewards (RLVR)

**Reinforcement Learning with Verifiable Rewards (RLVR)** is a training paradigm designed to align Large Language Models (LLMs) and other autonomous agents using objective, externally verifiable signals. The core problem RLVR addresses is the need for a powerful and trustworthy alignment mechanism that replaces subjective reward models with ground-truth, tamper-proof feedback to ensure precision in accuracy-critical domains.

### Method and Recipe

The RLVR process follows a systematic loop to iteratively improve a policy model based on deterministic verification. The step-by-step recipe is as follows:

1.  **Sampling**: Given a prompt $s$, the system draws one or more candidate completions $\{a\}_{1..k}$ from a policy model $\pi_\theta$.
2.  **Verification**: Each completion is processed by a deterministic function $r(s, \{a\})$ that checks for correctness. These verifiers may include unit tests, formal proofs, fact-checkers, compilers, or schema validators.
3.  **Rewarding**: The system assigns a binary reward based on the verification outcome:
    *   If the completion is verifiably correct, it receives a reward $r = \gamma$.
    *   If the completion is incorrect, it receives a reward $r = 0$.
4.  **Policy Update**: The policy parameters $\theta$ are updated using reinforcement learning algorithms (e.g., Proximal Policy Optimization, or PPO) to maximize the externally verifiable reward.
5.  **Verifier Refinement (Optional)**: The verifier itself may be trained, hardened, or expanded to encompass new edge cases, allowing the agent to self-bootstrap its learning signal.

### Key Technical Properties

RLVR is characterized by several distinct behavioral and structural advantages:
*   **Intrinsic Safety**: Every reward is traceable to a transparent verifier run, which simplifies compliance and debugging.
*   **Emergent Capabilities**: The use of sparse, high-precision rewards encourages systematic exploration, often leading to "aha-moments" where the model discovers a correct strategy, resulting in sudden surges in capability.
*   **Generalization**: Models trained on these verifiable objectives tend to extrapolate to unseen tasks with minimal additional data.
*   **Domain Agnosticism**: The framework is applicable across various fields, including theorem proving, code generation, robotics, and games.

### Quantitative Results and Scope

As a curated resource repository, the source does not provide a single experimental result set but documents the scale of the RLVR ecosystem. The collection includes:
*   **Extensive Literature**: The repository has integrated 135 papers from ICLR 2026 and ICML 2026.
*   **Implementation Frameworks**: The source identifies multiple high-performance codebases for RLVR, including **open-r1** (a reproduction of the DeepSeek-R1 pipeline), **OpenRLHF** (supporting PPO, GRPO, and REINFORCE++), and **verl**.

### Stated Limitations

While the source primarily highlights the strengths of RLVR, it identifies specific challenges and failure modes through its curated resources:
*   **Reward Hacking**: The "RLVR Book" resource specifically lists reward hacking as a key area of concern.
*   **Reasoning Failures**: Referenced tutorials note the existence of theoretical foundations and failure modes regarding LLM reasoning.
*   **Overthinking**: The source cites surveys regarding the "overthinking" phenomenon in reasoning models, suggesting a need for length-control techniques.
