---
id: github:awesome-rlvr-reinforcement-learning-with
type: web
title: Awesome RLVR — Reinforcement Learning with Verifiable Rewards
url: https://github.com/opendilab/awesome-RLVR
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

# Reinforcement Learning with Verifiable Rewards (RLVR)

**Reinforcement Learning with Verifiable Rewards (RLVR)** is an emerging alignment paradigm designed to optimize Large Language Models (LLMs) and other autonomous agents using objective, externally verifiable signals. Unlike traditional reinforcement learning from human feedback (RLHF), which may rely on subjective preference models, RLVR utilizes ground-truth feedback—such as unit tests, formal mathematical proofs, or fact-checkers—to provide binary, tamper-proof rewards.

### Core Problem
The primary challenge RLVR addresses is the need for trustworthy and precise alignment signals. Subjective reward models can be biased or susceptible to "reward hacking." RLVR seeks to replace these with objective verifiers to ensure intrinsic safety, auditability, and the ability for models to extrapolate to unseen tasks through systematic exploration. This approach is particularly critical for accuracy-critical domains like code generation, theorem proving, and robotics.

### Method/Recipe
The RLVR training process follows a cyclical loop to iteratively refine the policy model:

1.  **Sampling**: Given a prompt $s$, the system draws one or more candidate completions $\{a\}_{1..k}$ from the current policy model $\pi_\theta$.
2.  **Verification**: Each completion is passed through a deterministic verification function $r(s, \{a\})$. This function checks the completion for correctness using external tools (e.g., compilers, schema validators, or formal verifiers).
3.  **Rewarding**: The system assigns a reward based on the verification outcome:
    *   If the completion is verifiably correct: $r = \gamma$
    *   Otherwise: $r = 0$
4.  **Policy Update**: The policy parameters $\theta$ are updated using reinforcement learning algorithms (such as PPO or GRPO) to maximize the expected verifiable reward.
5.  **Verifier Refinement (Optional)**: The verifier may be hardened, expanded, or trained to cover new edge cases, creating a self-bootstrapping improvement loop.

### Key Formulas
The RLVR framework is characterized by the following technical components:
*   **Policy Model**: $\pi_\theta$
*   **Prompt**: $s$
*   **Candidate Completions**: $\{a\}_{1..k}$
*   **Reward Function**: 

$$
r(s, \{a\}) = 
\begin{cases} 
\gamma & \text{if verifiably correct} \\ 
0 & \text{otherwise} 
\end{cases}
$$

### Key Quantitative Results and Scope
As a curated resource repository rather than a single experimental study, the source provides quantitative data regarding the scale of the RLVR field rather than specific model benchmarks. Notable metrics include:
*   **Research Volume**: The collection includes 135 papers specifically from ICLR 2026 and ICML 2026.
*   **Implementation Ecosystem**: The paradigm is supported by numerous open-source codebases, including `open-r1` (a reproduction of DeepSeek-R1), `OpenRLHF` (supporting PPO, GRPO, and REINFORCE++), and `verl`.

### Stated Limitations
The provided text does not explicitly list a dedicated set of limitations for the RLVR paradigm. However, it references external tutorials (e.g., by Denny Zhou) that discuss the "failure modes of reasoning" in LLMs. The methodology implicitly requires the existence of a "deterministic function" for verification, suggesting that RLVR is limited to domains where such an objective verifier can be constructed.
